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
from typing import List, Dict, Any, Tuple
from abc import ABC, abstractmethod
import json
import torch
import cv2
import numpy as np
from fused_ssim import fused_ssim
from featup.featurizers.maskclip.clip import tokenize
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
from tqdm import tqdm
from PIL import Image

BACKGROUND_WORDS = [
    "floor",
    "walls",
    "ceiling",
    "background",
    "road",
    "sky",
    "table",
    "chair",
    "bed",
    "sofa",
    "cabinet",
    "shelf",
    "other",
]


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
    def __init__(self, label_folder: Path, rendered_folder: Path):
        super().__init__(label_folder, rendered_folder)
        self.tokenzier = tokenize
        self.text_encoder = torch.hub.load(
            "mhamilton723/FeatUp", "maskclip", use_norm=False
        ).model.model
        self.text_encoder.eval()
        self.text_encoder.to("cuda")
        self.background_text = BACKGROUND_WORDS

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

    def compute_metrics(self, save_path: Path) -> Dict[str, Any]:
        """
        Compute the metrics
        """
        # Store only per-frame metrics
        frame_names = []
        frame_metrics_list = []

        save_path.mkdir(parents=True, exist_ok=True)
        save_path_metrics_images = save_path / "metrics_images"
        save_path_metrics_images.mkdir(parents=True, exist_ok=True)

        for frame in tqdm(self.labels):
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
            feature_path = self.rendered_folder / "Feature" / f"{basename}.pt"
            if not feature_path.exists():
                raise FileNotFoundError(f"No feature file found for {feature_path}")
            rendered_feature = torch.load(str(feature_path)).cuda()

            feature_metrics = self.feature_metrics(
                rendered_feature, gt_masks_tensor, unique_categories
            )
            image_metrics = self.image_metrics(rendered_image, gt_image)

            # Store only the per-frame metrics
            frame_names.append(basename)
            frame_metrics_list.append(
                {
                    "frame_name/metrics_name": basename,
                    "mIoU": float(feature_metrics["mIoU"]),
                    "mAcc": float(feature_metrics["mAcc"]),
                    "SSIM": float(image_metrics["ssim"]),
                    "PSNR": float(image_metrics["psnr"])
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
        metrics_df = pd.DataFrame(frame_metrics_list)
        metrics_df.set_index('frame_name/metrics_name', inplace=True)

        # Compute per-scene mean
        scene_mean = metrics_df.mean()
        scene_mean.name = "mean"  # Set the index for the mean row

        # Append mean row to DataFrame
        scene_mean_df = scene_mean.to_frame().T
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
    ) -> Dict[str, Any]:
        """
        Compute the metrics for the feature
        """
        # 1. Get the text features
        tokens = self.tokenzier(unique_categories + self.background_text).to("cuda")
        text_features = self.text_encoder.encode_text(tokens).float().squeeze()
        feature_map_reshaped = rendered_feature.view(
            -1, rendered_feature.shape[-1]
        )  # Flatten spatial dims
        text_features = text_features / text_features.norm(
            dim=-1, keepdim=True
        )  # Normalize text features
        feature_map_reshaped = feature_map_reshaped / feature_map_reshaped.norm(
            dim=-1, keepdim=True
        )  # Normalize feature map
        # Compute similarity (logits)
        attention_scores = torch.matmul(
            feature_map_reshaped, text_features.T
        )  # Shape: (H*W, num_classes)
        predicted_labels = torch.argmax(
            attention_scores, dim=-1
        )  # Get highest similarity category index

        H, W, _ = rendered_feature.shape
        segmentation_masks = predicted_labels.view(H, W).cpu().numpy()

        #
        num_classes = text_features.shape[0]
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

    def mIoU(
        self, segmentation_masks: torch.Tensor, masks_tensor: torch.Tensor
    ) -> Tuple[List[float], float]:

        """
        Calculate the IoU for each class between the predicted dense segmentation and ground truth binary masks,
        and compute a weighted overall mIoU per frame based on the total number of pixels in each ground truth mask.

        Args:
            predicted_dense_masks (np.ndarray): 2D array where each pixel's value represents its predicted class label.
            gt_bindary_masks (List[np.ndarray]): List of binary masks (each of shape (H, W)) for each ground truth class.

        Returns:
            Tuple[List[float], float]: A tuple containing:
                - A list of IoU values for each class.
                - The weighted overall mIoU for the frame.
        """

        target_shape = masks_tensor[0].shape  # (height, width)

        # Check if resizing is needed
        if segmentation_masks.shape != target_shape:
            # cv2.resize expects size as (width, height) and uses nearest neighbor to avoid smoothing
            segmentation_masks = (
                torch.nn.functional.interpolate(
                    segmentation_masks.unsqueeze(0)
                    .unsqueeze(0)
                    .float(),  # Add batch and channel dims
                    size=(target_shape[0], target_shape[1]),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )  # Remove extra dims and convert back to long

        IoUs = []
        weights = []

        # Loop through each ground truth mask (each corresponding to a class)
        for i in range(len(masks_tensor)):
            # Create a binary mask for predicted segmentation for class i
            predicted_mask = segmentation_masks == i
            # Get the corresponding ground truth mask for class i
            gt_mask = masks_tensor[i]

            # Compute intersection and union for IoU calculation
            intersection = np.logical_and(predicted_mask, gt_mask).sum()
            union = np.logical_or(predicted_mask, gt_mask).sum()

            # Compute IoU and avoid division by zero
            iou = intersection / union if union > 0 else 0
            IoUs.append(iou)

            # Weight for the class: total number of pixels in the ground truth mask
            weights.append(gt_mask.sum())

        # Calculate overall mIoU as a weighted average using the ground truth pixel counts as weights
        total_weight = sum(weights)
        overall_miou = (
            (sum(iou * w for iou, w in zip(IoUs, weights)) / total_weight)
            if total_weight > 0
            else 0
        )

        # Print overall mIoU for the frame
        # print(f"Overall mIoU for the frame: {overall_miou:.4f}")

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
        segmentation_masks = metrics["segmentation_masks"]
        H, W = segmentation_masks.shape
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

        # Rendered segmentation mask (color)
        rendered_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(min(num_classes, np.max(segmentation_masks) + 1)):
            rendered_seg[segmentation_masks == i] = (colors[i][:3] * 255).astype(
                np.uint8
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
