"""
    We will use query words on the rendered images and features
    to compute the metrics.
    We need to first parser the json file
    And then, we can generate PSNR, SSIM, LPIPS, mIoU, ACC, F1, etc.

    The input is the following:
    - The rendered images
    - The rendered features
    - The query words, which is a list of strings
    - The ground truth labels, usually from json file

    The output is the following:
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple, Literal
from abc import ABC, abstractmethod
import json
import torch
import cv2
import numpy as np
from fused_ssim import fused_ssim
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from tqdm import tqdm
from PIL import Image
from text_encoder import TextEncoder
from sklearn.decomposition import PCA
import torch.nn.functional as F
import math
from evaluator_loader import BACKGROUND_WORDS



def pca_reduce(X: torch.Tensor, c_prime: int) -> torch.Tensor:
    """
    Perform PCA on X (n × c) to reduce it to shape (n × c_prime) using scikit-learn (CPU).

    Args:
        X:        Tensor of shape (n, c)
        c_prime:  Target number of principal components

    Returns:
        Tensor of shape (n, c_prime) containing the projected data.
    """
    # 1) Move to CPU and convert to NumPy
    X_np = X.detach().cpu().numpy()
    
    # 2) Fit PCA and transform
    pca = PCA(n_components=c_prime)
    X_reduced_np = pca.fit_transform(X_np)
    
    # 3) Convert back to torch.Tensor (on CPU)
    return torch.from_numpy(X_reduced_np).cuda()

def positional_embedding(
    positions: torch.Tensor, 
    x_max: int, 
    y_max: int, 
    num_freqs: int = 4
) -> torch.Tensor:
    """
    positions: (...,2) tensor of (x, y) pixel coords
    x_max, y_max: maximum x and y values for normalization
    num_freqs: number of frequency bands per axis;
               total output dim = 4 * num_freqs (→ 16 if num_freqs=4)
    returns: (..., 4*num_freqs) positional embedding
    """
    # normalize to [0,1]
    x = positions[..., 0] / x_max
    y = positions[..., 1] / y_max

    # create frequency bands: [1, 2, 4, 8, ...]
    freq_bands = 2.0 ** torch.arange(num_freqs, device=positions.device, dtype=positions.dtype)  # (num_freqs,)

    # angles = pos_norm * freq * π
    angles_x = x.unsqueeze(-1) * freq_bands * math.pi  # (..., num_freqs)
    angles_y = y.unsqueeze(-1) * freq_bands * math.pi  # (..., num_freqs)

    # stack sin & cos per axis
    pe_x = torch.cat([angles_x.sin(), angles_x.cos()], dim=-1)  # (..., 2*num_freqs)
    pe_y = torch.cat([angles_y.sin(), angles_y.cos()], dim=-1)  # (..., 2*num_freqs)

    # final embedding: [sin_x,cos_x, sin_y,cos_y] across freqs
    pe = torch.cat([pe_x, pe_y], dim=-1)                        # (..., 4*num_freqs)

    return pe


class Metrics(ABC):
    def __init__(self, label_folder: Path, rendered_folder: Path):
        self.label_folder = label_folder
        self.rendered_folder = rendered_folder
        self.labels = self.load_labels()

    @abstractmethod
    def load_labels(self) -> Dict[str, Any]:
        """
        Load the labels from the json file
        """
        pass

    @abstractmethod
    def compute_metrics(self, save_path: Path) -> Dict[str, Any]:
        """
        Compute the metrics
        """
        pass


class LERFMetrics(Metrics):
    def __init__(self, label_folder: Path, 
            rendered_folder: Path, 
            text_encoder: Literal["maskclip", "SAM2OpenCLIP", "SAMOpenCLIP"], 
            enable_pca: int|None = None,
            instance_segmentation: bool = True):
        super().__init__(label_folder, rendered_folder)
        self.text_encoder = TextEncoder(text_encoder, torch.device("cuda"))
        self.background_text = BACKGROUND_WORDS
        self.enable_pca = enable_pca
        self.position_embedding_weight = 0.001
        self.instance_segmentation = True

    def load_labels(self) -> Dict[str, Any]:
        """
        Load the labels from the json file
        """
        fileNames = list(self.label_folder.iterdir())
        fileNames.sort()
        # Data Structure: Scene Dict
        # {Scene_Name, Scene_Jsons, Scene_Images, Scene_Features}

        metadata = {}

        # First, collect all JSON files and their basenames
        json_files = [f for f in fileNames if str(f).endswith(".json")]
        for json_file in json_files:
            basename = Path(json_file).stem
            metadata[basename] = {"json": str(json_file)}

        # Then, match image files to their corresponding JSON files
        image_files = [
            f for f in fileNames if str(f).endswith((".jpg", ".png", ".jpeg"))
        ]
        for image_file in image_files:
            basename = Path(image_file).stem
            if basename in metadata:
                metadata[basename]["image"] = str(image_file)

        # Convert metadata to list of dictionaries
        result = {}
        for basename, data in metadata.items():
            if "json" in data and "image" in data:  # Only include complete pairs
                result[basename] = {
                    "json_path": data["json"],
                    "image_path": data["image"],
                }

        return result

    def compute_metrics(self, save_path: Path, mode: Literal["feature_map", "attention_map"]) -> Dict[str, Any]:
        """
        Compute the metrics
        """
        # Store only per-frame metrics
        frame_names = []
        frame_metrics_list = []

        save_path.mkdir(parents=True, exist_ok=True)
        save_path_metrics_images = save_path / "metrics_images"
        save_path_metrics_images.mkdir(parents=True, exist_ok=True)

        for frame in tqdm(self.labels, desc="Computing metrics"):
            json_path = self.labels[frame]["json_path"]
            gt_image_path = Path(self.labels[frame]["image_path"])
            gt_image = cv2.imread(str(gt_image_path))
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_image = torch.tensor(gt_image)
            gt_masks_tensor, unique_categories = self.json_parser(json_path)

            # Get basename without extension
            basename = gt_image_path.stem

            # Try different image formats
            rendered_image = None
            img_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = self.rendered_folder / "RGB" / f"{basename}{ext}"
                if img_path.exists():
                    rendered_image = cv2.imread(str(img_path))
                    rendered_image = cv2.cvtColor(rendered_image, cv2.COLOR_BGR2RGB)
                    rendered_image = torch.tensor(rendered_image)
                    break
            if rendered_image is None:
                raise FileNotFoundError(f"No rendered image found for {str(img_path)}")

            # Load feature file
            if mode == "feature_map":
                feature_path = self.rendered_folder / "Feature" / f"{basename}.pt"
            elif mode == "attention_map" or mode == "drsplat":
                feature_path = self.rendered_folder / "AttentionMap" / f"{basename}.pt"
            else:
                raise ValueError(f"Invalid mode: {mode}")
            
            if not feature_path.exists():
                raise FileNotFoundError(f"No feature file found for {feature_path}")
            rendered_feature = torch.load(str(feature_path)).cuda()

            feature_metrics = self.feature_metrics(
                rendered_feature, gt_masks_tensor, unique_categories,
                mode
            )
            image_metrics = self.image_metrics(rendered_image, gt_image)

            # Store only the per-frame metrics
            frame_names.append(basename)
            frame_metrics_list.append(
                {
                    "mIoU": float(feature_metrics["mIoU"]),
                    "mAcc": float(feature_metrics["mAcc"]),
                    "SSIM": float(image_metrics["ssim"]),
                    "PSNR": float(image_metrics["psnr"]),
                }
            )

            metrics_images = self.visualize_metrics(
                feature_metrics,
                rendered_image,
                gt_masks_tensor,
                gt_image.cpu().numpy(),
                unique_categories,
            )
            # Convert tensor to PIL Image and save
            metrics_images = Image.fromarray(
                metrics_images.cpu().numpy().astype(np.uint8)
            )
            metrics_images.save(save_path / "metrics_images" / f"{basename}.png")

        # Create DataFrame for per-frame metrics
        metrics_df = pd.DataFrame(frame_metrics_list, index=frame_names)

        # Compute per-scene mean
        scene_mean = metrics_df.mean()
        scene_mean.name = "mean"  # Set the index for the mean row

        # Append mean row to DataFrame
        scene_mean_df = scene_mean.to_frame().T
        scene_mean_df.index = pd.Index(["mean"])
        metrics_df = pd.concat([metrics_df, scene_mean_df])

        # Save DataFrame with mean row
        metrics_df.to_csv(save_path / "frame_metrics.csv")

        return {"frame_metrics": metrics_df, "scene_mean": scene_mean}

    def json_parser(self, json_path: Path) -> Tuple[torch.Tensor, List[str]]:
        """
        Parse a JSON dictionary containing segmentation annotations and return:
          - A torch.Tensor of shape (num_categories, H, W) with binary masks aggregated by category
          - A list of unique category names.
        """
        json_dict = json.load(open(json_path, "r"))
        aggregated_masks = {}
        unique_categories = []

        W = json_dict["info"]["width"]
        H = json_dict["info"]["height"]

        for obj in json_dict["objects"]:
            category = obj["category"]
            seg = obj["segmentation"]

            # Create a binary mask with shape (H, W)
            mask = np.zeros((H, W), dtype=np.uint8)

            # Convert polygon points to a numpy array of shape (num_points, 1, 2)
            polygon_points = np.array(
                [[point[0], point[1]] for point in seg], dtype=np.int32
            ).reshape((-1, 1, 2))

            # Fill the polygon on the mask (the region inside becomes 1)
            cv2.fillPoly(mask, [polygon_points], (1,))

            if category in aggregated_masks:
                aggregated_masks[category] = np.maximum(
                    aggregated_masks[category], mask
                )
            else:
                aggregated_masks[category] = mask
                unique_categories.append(category)

        # Stack masks into a single torch.Tensor with shape (num_categories, H, W)
        masks_tensor = torch.from_numpy(
            np.stack([aggregated_masks[cat] for cat in unique_categories])
        )

        return masks_tensor, unique_categories

    def image_metrics(
        self, rendered_image: torch.Tensor, ground_truth_image: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Compute the PSNR between the rendered image and the ground truth image
        """
        mse = torch.nn.functional.mse_loss(
            rendered_image.float().cuda(), ground_truth_image.float().cuda()
        )
        psnr = float(10 * torch.log10(mse))
        ssim = fused_ssim(
            rendered_image.float().cuda().unsqueeze(0),
            ground_truth_image.float().cuda().unsqueeze(0),
        )
        metrics = {"psnr": psnr, "ssim": ssim}
        return metrics

    def feature_metrics(
        self,
        rendered_feature: torch.Tensor,
        masks_tensor: torch.Tensor,
        unique_categories: List[str],
        mode: Literal["feature_map", "attention_map", "drsplat"]
    ) -> Dict[str, Any]:
        """
        Compute the metrics for the feature
        """

        if mode == "feature_map":
            predicted_labels, attention_scores = self.feature_map2labels(rendered_feature, unique_categories)
            H, W, _ = rendered_feature.shape
        elif mode == "attention_map":
            predicted_labels, attention_scores = self.attention_map2labels(rendered_feature, unique_categories)
            H, W, _= rendered_feature.shape
        elif mode == "drsplat":
            predicted_labels, attention_scores = self.drsplat2labels(rendered_feature, unique_categories)
            H, W, _= rendered_feature.shape
        else:
            raise ValueError(f"Invalid mode: {mode}")

        segmentation_masks = predicted_labels.view(-1, H, W)

        num_classes = len(unique_categories)
        max_coords = []
        for cls in range(num_classes):
            class_attention = attention_scores[:, cls]  # (H*W,)
            # Get the top_n indices for the current class (sorted in descending order)
            topk = torch.topk(class_attention, k=1, sorted=True)
            coords = []
            for idx in topk.indices:
                y, x = divmod(idx.item(), W)  # Convert flat idx to 2D coord (y, x)
                coords.append((x, y))
            max_coords.append(coords)

        mIoU_per_objects, mIoU = self.mIoU(segmentation_masks, masks_tensor)
        acc_per_objects, mAcc = self.localization((H, W), max_coords, masks_tensor)

        metrics = {}
        for i in range(len(unique_categories)):
            metrics[unique_categories[i]] = {
                "mIoU": mIoU_per_objects[i],
                "acc": acc_per_objects[i],
            }

        metrics["mIoU"] = mIoU
        metrics["mAcc"] = mAcc
        metrics["segmentation_masks"] = segmentation_masks

        return metrics

    def drsplat2labels(self, attn_scores: torch.Tensor, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        In their code, it use 4 background words, and compare with threshold 0.5,
        0.5 means higher than any of the backgournd words, while do not compare with other words.
    
        Args:
          attn_scores: (H, W, B) float tensor of attention scores.
          prompts:     list of length B
          threshold:   minimum attention to keep a one-hot “1”; below → 0.
    
        Returns:
          mask:        (B, H, W) LongTensor with 0/1
          attn_scores: unchanged input, for downstream use
        """
        positive_attention_score = attn_scores[..., :len(prompts)]
        background_attention_score = attn_scores[..., len(prompts):]
        threshold = background_attention_score.max(dim=-1).values # 0.5 means simple comparison with background words


        masks = positive_attention_score > threshold[..., None]
        masks = masks.permute(2, 0, 1).long()

        return masks, attn_scores
    
    def feature_map2labels(self, feature_map: torch.Tensor, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert the feature map to labels, as well as the attention scores
        feature_map: (H, W, C)
        texts: (B, ) List[str]
        return: (B, H, W) torch.Tensor dense segmentation masks, (B,) torch.Tensor attention scores
        """
        texts_query = texts + self.background_text
        text_features = self.text_encoder.encode_text(texts_query).float().squeeze()
        attention_scores = self.compute_attention_with_positional_embedding(feature_map, text_features)

        predicted_labels = torch.argmax(attention_scores, dim=-1)  # Get highest similarity category index

        num_classes = int(predicted_labels.max().item()) + 1
        # one_hot: (H, W, B)
        one_hot = F.one_hot(predicted_labels, num_classes=num_classes)
        
        # reorder to (B, H, W)
        one_hot = one_hot.permute(2, 0, 1).contiguous()

        return one_hot[:len(texts)], attention_scores

    @torch.no_grad()
    def attention_map2labels(    
        self,
        attn_scores: torch.Tensor,
        prompts: List[str],
        threshold: float = 0.85,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert an (H, W, B) attention map into a (B, H, W) one-hot mask,
        then zero-out any pixel whose attention score < threshold.
    
        Args:
          attn_scores: (H, W, B) float tensor of attention scores.
          prompts:     list of length B
          threshold:   minimum attention to keep a one-hot “1”; below → 0.
    
        Returns:
          mask:        (B, H, W) LongTensor with 0/1
          attn_scores: unchanged input, for downstream use
        """
        # 1) find the top class per pixel
        predicted = torch.argmax(attn_scores, dim=-1)  # (H, W)
    
        B = len(prompts) + len(BACKGROUND_WORDS)
        # 2) one-hot encode into (H, W, B), then permute → (B, H, W)
        one_hot = F.one_hot(predicted, num_classes=B)      # (H, W, B)
        one_hot = one_hot.permute(2, 0, 1).contiguous()    # (B, H, W)
    
        # 3) build a threshold mask: True where attn >= threshold
        #    need to align dims: (H, W, B) → (B, H, W)
        thresh_mask = (attn_scores >= threshold).permute(2, 0, 1)  # bool (B, H, W)
    
        # 4) combine: only keep one_hot==1 where thresh_mask is True
        mask = (one_hot.bool() & thresh_mask).long()  # (B, H, W), dtype=torch.int64
    
        return mask[:len(prompts)], attn_scores
    

    @torch.no_grad()
    def attention_map2threshold(
        self,
        attn_scores: torch.Tensor,
        prompts: List[str],
        threshold: float = 0.9,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert an (H, W, B) attention map into a (B, H, W) multi‐hot mask
        by directly thresholding each channel.

        Args:
          attn_scores: (H, W, B) float tensor of attention scores.
          prompts:     list of length B'  (we’ll ignore any extra background channels)
          threshold:   cutoff; any score >= threshold becomes 1

        Returns:
          mask:        (B', H, W) LongTensor with 0/1
          attn_scores: unchanged input
        """
        H, W, B = attn_scores.shape
        total_B = len(prompts) + len(BACKGROUND_WORDS)
        assert B == total_B, "Expected exactly B'+|BG| channels"

        # 1) direct per‐channel threshold → bool mask (H, W, B)
        bool_map = attn_scores >= threshold  # True wherever score >= threshold

        # 2) permute to (B, H, W) and cast to int
        mask = bool_map.permute(2, 0, 1).long()  # (B, H, W)

        # 3) drop any background channels if you want only the prompt ones
        return mask[:len(prompts)], attn_scores

    def compute_attention_with_positional_embedding(
        self,
        features_hw_c: torch.Tensor,   # (H, W, C)
        text_feats_b_c: torch.Tensor,  # (B, C)
    ) -> torch.Tensor:                 # → (H, W, B)
        H, W, C = features_hw_c.shape
        B, _     = text_feats_b_c.shape

        # 1) normalize
        f = F.normalize(features_hw_c, dim=-1)   # (H,W,C)
        t = F.normalize(text_feats_b_c, dim=-1)  # (B,C)

        # 2) standard attention score
        attn = torch.einsum('hwc,bc->hwb', f, t)  # (H,W,B)

        # 3) find the argmax location for each text‐class b
        flat   = attn.view(-1, B)                # (H*W, B)
        idx    = flat.argmax(dim=0)              # (B,) flat indices
        ys     = idx // W
        xs     = idx %  W
        pos_b2 = torch.stack([xs, ys], dim=-1).float().to(features_hw_c.device)  # (B,2)

        # 4) build the positional grid for all pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=features_hw_c.device),
            torch.arange(W, device=features_hw_c.device),
            indexing='ij'
        )
        pos_hw2 = torch.stack([x_coords, y_coords], dim=-1).float()  # (H, W, 2)

        # 5) compute embeddings
        pos_emb_hw = positional_embedding(pos_hw2, x_max=W-1, y_max=H-1)  # (H, W, 4)
        pos_emb_b  = positional_embedding(pos_b2,  x_max=W-1, y_max=H-1)  # (B, 4)

        # 6) positional affinity score between each pixel and each class‐peak
        pos_score = torch.einsum('hwe,be->hwb', pos_emb_hw, pos_emb_b)    # (H, W, B)

        # 7) combine
        return attn + self.position_embedding_weight * pos_score     


    def mIoU(
        self, segmentation_masks: torch.Tensor, masks_tensor: torch.Tensor, weighted: bool = False
    ) -> Tuple[List[float], float]:

        """
        Calculate the IoU for each class between the predicted dense segmentation and ground truth binary masks,
        and compute a weighted overall mIoU per frame based on the total number of pixels in each ground truth mask.

        Args:
            predicted_dense_masks (torch.Tensor): (B, H, W) tensor where each pixel's value represents its predicted class label.
            gt_bindary_masks (torch.Tensor): (B, H, W) tensor where each pixel's value represents its ground truth class label.
            weighted (bool): Whether to use weighted mIoU. 
            Some people use the number of pixels in the ground truth mask as the weight,
            some people use the number of pixels in the predicted mask as the weight.
            Some don't use weighted mIoU. 

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - A list of IoU values for each class.
                - The weighted overall mIoU for the frame.
        """

        H,W = masks_tensor[0].shape  # (height, width)
        B, H1, W1 = segmentation_masks.shape
        assert H == H1 and W == W1, "The shape of the predicted and ground truth masks must be the same"


        IoUs = []
        weights = []

        # Loop through each ground truth mask (each corresponding to a class)
        for i in range(len(masks_tensor)):
            # Create a binary mask for predicted segmentation for class i
            # Get the corresponding ground truth mask for class i
            gt_mask = masks_tensor[i]

            # Compute intersection and union for IoU calculation
            gt_mask = gt_mask.cuda()
            intersection = torch.logical_and(segmentation_masks[i], gt_mask).sum()
            union = torch.logical_or(segmentation_masks[i], gt_mask).sum()

            # Compute IoU and avoid division by zero
            iou = intersection / union if union > 0 else 0
            IoUs.append(float(iou))

            # Weight for the class: total number of pixels in the ground truth mask
            weights.append(float(gt_mask.sum()))

        # Calculate overall mIoU as a weighted average using the ground truth pixel counts as weights
        total_weight = sum(weights)
        if weighted:    
            overall_miou = (
                (sum(iou * w for iou, w in zip(IoUs, weights)) / total_weight)
                if total_weight > 0
                else 0
            )
        else:
            overall_miou = sum(IoUs) / len(IoUs)

        return IoUs, overall_miou

    # LeRF Localization
    def localization(
        self,
        orig_shape: Tuple[int, int],
        max_coordinates: List[List[Tuple[int, int]]],
        masks_tensor: torch.Tensor,
    ) -> Tuple[List[float], float]:
        """
        Calculate the localization accuracy for each class by checking if any of the top n coordinates
        is inside the corresponding ground truth binary mask. Adjust the max_coordinates to match the
        resolution of the ground truth masks if needed, and compute the mean accuracy across classes.

        Args:
            orig_shape (tuple): The original shape (height, width) of the predicted mask.
            max_coordinates (List[List[tuple]]): List of lists of (x, y) coordinates for each class,
                where each sublist is sorted with the highest attention scores first.
            gt_bindary_masks (List[np.ndarray]): List of binary masks (each of shape (H, W)) for each ground truth class.
            n (int): The number of top coordinates to consider for each class.

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - A list of localization accuracy values for each class (1 if any of the top n coordinates is within
                  the ground truth mask, else 0).
                - The mean localization accuracy across classes.
        """
        target_shape = masks_tensor[0].shape  # (height, width)

        # Adjust the coordinates if the original shape differs from target shape.
        if orig_shape != target_shape:
            scale_x = target_shape[1] / orig_shape[1]
            scale_y = target_shape[0] / orig_shape[0]
            adjusted_coordinates = []
            for coords in max_coordinates:
                # Take only the top n coordinates and adjust them.
                adjusted = [
                    (int(round(x * scale_x)), int(round(y * scale_y)))
                    for (x, y) in coords
                ]
                adjusted_coordinates.append(adjusted)
            max_coordinates = adjusted_coordinates

        localization_acc = []
        H, W = target_shape

        # Iterate over each class (only up to the number of available ground truth masks).
        for i, coords in enumerate(max_coordinates[: len(masks_tensor)]):
            found = False
            for (x, y) in coords:
                # Check that the coordinate is within bounds.
                if x < 0 or x >= W or y < 0 or y >= H:
                    continue
                # If any coordinate is inside the ground truth mask (i.e. nonzero), mark as correct.
                if masks_tensor[i][y, x]:
                    found = True
                    break
            localization_acc.append(1.0 if found else 0.0)

        mean_acc = (
            sum(localization_acc) / len(localization_acc) if localization_acc else 0.0
        )

        return localization_acc, mean_acc

    def visualize_metrics(
        self,
        metrics: Dict[str, Any],
        rendered_image: torch.Tensor,
        masks_tensor: torch.Tensor,
        ground_truth_image: np.ndarray,
        unique_category_text: List[str],
    ) -> torch.Tensor:
        """
        Visualize metrics with a 2x2 grid:
        [Rendered image | GT image]
        [Rendered segmentation | GT segmentation]
        """
        segmentation_masks = metrics["segmentation_masks"] # Notice that the shape is (B, H, W), sparse mask
        B, H, W = segmentation_masks.shape
        num_classes = len(unique_category_text)
        cmap = plt.cm.get_cmap("tab20", num_classes)
        colors = np.array([cmap(i) for i in range(num_classes)])  # RGBA

        # Prepare images
        rendered_np = rendered_image.cpu().numpy()
        rendered_np = (rendered_np - rendered_np.min()) / (
            rendered_np.max() - rendered_np.min()
        )
        rendered_np = (rendered_np * 255).astype(np.uint8)
        if rendered_np.shape[0] == 3:
            rendered_np = np.transpose(rendered_np, (1, 2, 0))

        gt_np = ground_truth_image
        if gt_np.max() <= 1.0:
            gt_np = (gt_np * 255).astype(np.uint8)
        if gt_np.shape[0] == 3:
            gt_np = np.transpose(gt_np, (1, 2, 0))

        rendered_seg = _colorize_sparse_masks(
            segmentation_masks,   # your sparse (B,H,W)
            colors,               # RGBA colormap array
            alpha=1
        )

        # GT segmentation mask (color)
        gt_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(masks_tensor.shape[0]):
            mask = masks_tensor[i].cpu().numpy()
            gt_seg[mask == 1] = (colors[i][:3] * 255).astype(np.uint8)

        # Plot
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs[0, 0].imshow(rendered_np)
        axs[0, 0].set_title("Rendered Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(gt_np)
        axs[0, 1].set_title("Ground Truth Image")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(rendered_seg)
        axs[1, 0].set_title("Rendered Segmentation")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(gt_seg)
        axs[1, 1].set_title("Ground Truth Segmentation")
        axs[1, 1].axis("off")

        # Add legend below
        from matplotlib.patches import Patch

        legend_patches = [
            Patch(color=colors[i][:3], label=unique_category_text[i])
            for i in range(num_classes)
        ]
        fig.legend(
            handles=legend_patches,
            loc="lower center",
            ncol=4,
            frameon=True,
            fontsize=10,
        )

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)

        # Save to temporary file and convert to tensor
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name, bbox_inches="tight", pad_inches=0)
            plt.close(fig)
            saved_img = Image.open(tmp.name).convert("RGB")
            result = torch.tensor(np.array(saved_img)).float()

        return result



def _colorize_sparse_masks(
    sparse_masks,        # torch sparse_coo_tensor or dense Tensor, shape (B,H,W)
    colors,              # np.ndarray of shape (num_classes,4) RGBA
    alpha: float = 0.3
) -> np.ndarray:
    """
    Returns an (H, W, 3) uint8 RGB image where each class b
    from sparse_masks[b] is overlaid with colors[b] and opacity alpha.
    Overlaps naturally blend in sequence.
    """
    # bring to dense numpy shape (B,H,W)
    if sparse_masks.is_sparse:
        mask_np = sparse_masks.to_dense().cpu().numpy()
    else:
        mask_np = sparse_masks.cpu().numpy()

    B, H, W = mask_np.shape
    # start with black background in float
    canvas = np.zeros((H, W, 3), dtype=np.float32)

    for b in range(B):
        # boolean mask for class b
        m = mask_np[b].astype(bool)  # (H,W)
        if not m.any():  
            continue
        color = colors[b, :3]  # RGB in [0,1]
        # blend: new = old*(1-alpha) + color*alpha
        canvas[m] = canvas[m] * (1 - alpha) + color * alpha

    # convert to uint8
    canvas = (canvas * 255).clip(0,255).astype(np.uint8)
    return canvas