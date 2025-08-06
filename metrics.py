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
from typing import List, Dict, Any, Tuple, Literal, Union
from abc import ABC, abstractmethod
import json
from sklearn.decomposition import PCA
import torch
import cv2
import numpy as np
from fused_ssim import fused_ssim
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from PIL import Image
from gsplat_ext import TextEncoder
import torch.nn.functional as F
from evaluator_loader import BACKGROUND_WORDS
from scipy import ndimage
import numpy as np



class Metrics(ABC):
    def __init__(self, label_folder: Path, rendered_folder: Path, feature_folder: Union[Path, None] = None):
        self.label_folder = label_folder
        self.rendered_folder = rendered_folder
        self.labels = self.load_labels(feature_folder)

    @abstractmethod
    def load_labels(self, feature_folder: Union[Path, None] = None) -> Dict[str, Any]:
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
    def __init__(
        self,
        label_folder: Path,
        rendered_folder: Path,
        text_encoder: Literal["maskclip", "SAM2OpenCLIP", "SAMOpenCLIP"],
        enable_pca: int | None = None,
        instance_segmentation: bool = True,
        feature_folder: Union[Path, None] = None,
    ):
        super().__init__(label_folder, rendered_folder, feature_folder)
        #self.text_encoder = TextEncoder(text_encoder, torch.device("cuda"))
        self.background_text = BACKGROUND_WORDS
        self.enable_pca = enable_pca
        self.position_embedding_weight = 0.001
        self.instance_segmentation = True

    def load_labels(self, feature_folder: Union[Path, None] = None) -> Dict[str, Any]:
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

        if feature_folder is not None:
            feature_files = [f for f in feature_folder.iterdir() if str(f).endswith(".pt")]
            for feature_file in feature_files:
                basename = Path(feature_file).stem
                if basename in metadata:
                    metadata[basename]["feature"] = str(feature_file)

        # Convert metadata to list of dictionaries
        result = {}
        for basename, data in metadata.items():
            if "json" in data and "image" in data:  # Only include complete pairs
                result[basename] = {
                    "json_path": data["json"],
                    "image_path": data["image"],
                    "feature_path": data["feature"] if feature_folder is not None else None
                }

        return result

    def compute_metrics(
        self, save_path: Path, mode: Literal["feature_map", "attention_map"], feature_folder: Union[Path, None] = None
    ) -> Dict[str, Any]:
        """
        Compute the metrics
        """
        # Store only per-frame metrics
        frame_names = []
        frame_metrics_list = []


        if mode == "feature_map":
            assert feature_folder is not None, "feature_folder is required for feature_map"
        
        save_path.mkdir(parents=True, exist_ok=True)
        save_path_metrics_images = save_path / "metrics_images"
        save_path_metrics_images.mkdir(parents=True, exist_ok=True)

        for frame in tqdm(self.labels, desc="Computing metrics"):
            json_path = self.labels[frame]["json_path"]
            gt_image_path = Path(self.labels[frame]["image_path"])
            gt_image = cv2.imread(str(gt_image_path))
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)
            gt_image = torch.tensor(gt_image)
            if mode == "feature_map":
                gt_feature = torch.load(self.labels[frame]["feature_path"])
                gt_feature = gt_feature.to("cuda")
            else:
                gt_feature = None
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
                rendered_feature, gt_feature if mode == "feature_map" else gt_masks_tensor, unique_categories, mode
            )
            if mode == "feature_map":
                gt_masks_tensor = feature_metrics["gt_feature"].to("cuda")
            image_metrics = self.image_metrics(rendered_image, gt_image)

            # Store only the per-frame metrics
            frame_names.append(basename)
            if mode == "feature_map":
                frame_metrics = {
                    "cosine_similarity": float(feature_metrics["cosine_similarity"]),
                    "ssim": float(image_metrics["ssim"]),
                    "psnr": float(image_metrics["psnr"]),
                }
            else:
                frame_metrics = {
                    "mIoU": feature_metrics["mIoU"],
                    "mAcc": feature_metrics["mAcc"],
                    "ssim": float(image_metrics["ssim"]),
                    "psnr": float(image_metrics["psnr"]),
                }
            frame_metrics_list.append(
                frame_metrics
            )

            metrics_images = self.visualize_metrics(
                mode,
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
        unique_categories = {}

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
                unique_categories[category] += 1
            else:
                aggregated_masks[category] = mask
                unique_categories[category] = 1

        # Stack masks into a single torch.Tensor with shape (num_categories, H, W)
        unique_categories = list(unique_categories.items())
        masks_tensor = torch.from_numpy(
            np.stack([aggregated_masks[cat] for cat, _ in unique_categories])
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


    def morph_smooth(self, masks: torch.Tensor, k: int = 3, iters: int = 1) -> torch.Tensor:
        """
        masks: (B, H, W) binary {0,1}
        k: kernel size (odd)
        iters: how many times to apply opening+closing
        """
        assert masks.dim() == 3
        B, H, W = masks.shape
        x = masks.unsqueeze(1).float()  # (B,1,H,W)
        pad = k // 2
        weight = torch.ones(1, 1, k, k, device=x.device)

        for _ in range(iters):
            # Erode: keep pixel if all neighbors ==1
            erode = (F.conv2d(x, weight, padding=pad) == k*k).float()
            # Dilate: pixel becomes 1 if any neighbor ==1
            dilate = (F.conv2d(erode, weight, padding=pad) > 0).float()
            # Opening done (erode->dilate) removes small bright noise
            x = dilate
            # Closing: dilate then erode (fills small dark holes)
            dil = (F.conv2d(x, weight, padding=pad) > 0).float()
            x = (F.conv2d(dil, weight, padding=pad) == k*k).float()

        return x.squeeze(1)


    def feature_metrics(
        self,
        rendered_feature: torch.Tensor,
        masks_tensor: torch.Tensor,
        unique_categories: List[str],
        mode: Literal["feature_map", "attention_map", "drsplat"],
    ) -> Dict[str, Any]:
        """
        Compute the metrics for the feature
        """
        unique_categories_str = [cat for cat, _ in unique_categories]
        if mode == "feature_map":
            cosine_similarity, rendered_feature, masks_tensor = self.compute_cosine_similarity_map(
                rendered_feature, masks_tensor
            )
            H, W, _ = rendered_feature.shape
            metrics = {"cosine_similarity": cosine_similarity, "feature_map": rendered_feature, "gt_feature": masks_tensor}
            return metrics
        elif mode == "attention_map":
            predicted_labels, attention_scores = self.attention_map2auto_threshold(
                rendered_feature, unique_categories
            )
            H, W, _ = rendered_feature.shape
        elif mode == "drsplat":
            predicted_labels, attention_scores = self.drsplat2labels(
                rendered_feature, unique_categories_str
            )
            H, W, _ = rendered_feature.shape
        else:
            raise ValueError(f"Invalid mode: {mode}")

        segmentation_masks = predicted_labels.view(-1, H, W)
        segmentation_masks = self.morph_smooth(segmentation_masks, k=3, iters=1)
        segmentation_masks = self.keep_largest_component_fill_holes(segmentation_masks)
        
        num_classes = len(unique_categories)
        max_coords = []
        for cls in range(num_classes):
            class_attention = attention_scores[..., cls].reshape(-1)  # (H*W,)
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

    def drsplat2labels(
        self, attn_scores: torch.Tensor, prompts: List[str]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        positive_attention_score = attn_scores[..., : len(prompts)]
        background_attention_score = attn_scores[..., len(prompts) :]
        threshold = background_attention_score.max(
            dim=-1
        ).values  # 0.5 means simple comparison with background words

        masks = positive_attention_score > threshold[..., None]
        masks = masks.permute(2, 0, 1).long()

        return masks, attn_scores

    def compute_cosine_similarity_map(
        self, feature_map: torch.Tensor, gt_feature_map: torch.Tensor
    ) -> Tuple[float, torch.Tensor, torch.Tensor]:
        """
        Compute mean cosine similarity between the feature map and the ground truth feature map.
        Assumes both feature_map and gt_feature_map are of shape (H, W, C) and L2-normalized.
        Returns: mean_cosine_sim, feature_map, gt_feature_map
        """
        # L2 normalization
        
        # Interpolate feature_map to match gt_feature_map's height and width
        if feature_map.shape[:2] != gt_feature_map.shape[:2]:
            feature_map = feature_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
            feature_map = torch.nn.functional.interpolate(
                feature_map,
                size=gt_feature_map.shape[:2],
                mode="bilinear",
                align_corners=False,
            )
            feature_map = feature_map.squeeze(0).permute(1, 2, 0)  # (H, W, C)
        feature_map = F.normalize(feature_map, p=2, dim=-1)
        gt_feature_map = F.normalize(gt_feature_map, p=2, dim=-1)

        # Cosine similarity: dot product between unit vectors
        cos_sim_map = torch.sum(feature_map * gt_feature_map, dim=-1)  # shape (H, W)
        mean_cos_sim = torch.mean(cos_sim_map)  # scalar

        return mean_cos_sim.item(), feature_map, gt_feature_map

    @torch.no_grad()
    def attention_map2auto_threshold(
        self,
        attn_scores: torch.Tensor,   # (H, W, B)
        prompts: List[Tuple[str, int]],
        min_bin: int = 40,
        min_peak_count: int = 600,
        use_rescale: bool = False,
        percentile_fallback: float = 0.75,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Auto-threshold each channel by histogram peak→valley heuristic.

        Returns:
            mask:          (B', H, W) binary (0/1) LongTensor for prompt channels
            thresholds:    (B,) float tensor of chosen thresholds in *original (or rescaled) range*
            rescaled_map:  (H, W, B) float tensor actually thresholded (may be same as input if use_rescale=False)
        """
        H, W, B= attn_scores.shape
        total_B = len(prompts)  + len(BACKGROUND_WORDS)
        assert B == total_B, f"Expected {total_B} channels, got {B}"

        x = attn_scores

        # Optional per-channel rescale (as in your original code)
        if use_rescale:
            flat = x.view(-1, B)                      # (H*W, B)
            means = flat.mean(dim=0).view(1, 1, B)
            maxs  = flat.max(dim=0).values.view(1, 1, B)
            denom = (maxs - means).clamp_min(1e-8)
            rescaled = ((x - means) / denom).clamp(0.0, 1.0)
        else:
            rescaled = x

        # Prepare output threshold tensor
        thresholds = torch.zeros(B, device=attn_scores.device, dtype=attn_scores.dtype)

        # We’ll do histogram on CPU for simplicity (speed fine for typical H,W)
        # If needed, keep on GPU and use torch.histc in a loop.
        res_cpu = rescaled.detach().cpu()

        def compute_hist(channel_vals: torch.Tensor, bins: int):
            # channel_vals: (H,W) -> flatten
            v = channel_vals.reshape(-1)
            # We assume values are in [0,1] if rescaled; otherwise compute min/max
            vmin = 0.0 if use_rescale else float(v.min())
            vmax = 1.0 if use_rescale else float(v.max())
            if vmax <= vmin + 1e-12:
                # Degenerate; return single bin
                counts = torch.tensor([v.numel()], dtype=torch.long)
                edges = torch.linspace(vmin, vmax + 1e-6, steps=2)
                return counts, edges
            # torch.histc gives counts over evenly spaced bins (excluding rightmost edge)
            counts = torch.histc(v, bins=bins, min=vmin, max=vmax)
            # Build edges to match (bins+1)
            edges = torch.linspace(vmin, vmax, steps=bins + 1)
            return counts.long(), edges

        def find_peak_valley(counts: torch.Tensor, instance_count: int):
            """
            counts: (bins,) long
            Returns: (peak_idx, valley_idx or None)
            Peak: local max; Valley: first local min to right with lower count.
            """
            bins = counts.numel()
            # Identify local maxima
            # For interior bins: c[i] >= c[i-1] and c[i] >= c[i+1]
            # Treat edges carefully
            local_max = []
            for i in range(bins):
                left  = counts[i-1] if i-1 >= 0 else counts[i]
                right = counts[i+1] if i+1 < bins else counts[i]
                if counts[i] >= left and counts[i] >= right:
                    local_max.append(i)
            if not local_max:
                return None, None
            # Filter by min_peak_count
            local_max = [i for i in local_max if counts[i] >= min_peak_count]
            if not local_max:
                return None, None
            # Largest peak (argmax count); tie → earliest
            if len(local_max) < instance_count:
                peak_idx = local_max[-1]
            else:
                peak_idx = local_max[-instance_count]

            # Find valley to the right
            peak_height = counts[peak_idx]
            valley_idx = None
            for j in range(peak_idx, 0, -1):
                # local minimum
                if counts[j] <= counts[j-1] and counts[j] <= counts[j+1] and counts[j] < peak_height:
                    valley_idx = j
                    break
            # If no interior valley, optionally take last decreasing point
            if valley_idx is None:
                # find last index where counts strictly lower than peak
                for j in range(bins - 1, peak_idx, -1):
                    if counts[j] < peak_height:
                        valley_idx = j
                        break
            return peak_idx, valley_idx

        for b in range(len(prompts)):
            channel = res_cpu[..., b]
            counts, edges = compute_hist(channel, bins=min_bin)
            
            peak_idx, valley_idx = find_peak_valley(counts, prompts[b][1])
            if peak_idx is not None and valley_idx is not None:
                # Threshold at midpoint of valley bin
                left_edge  = edges[valley_idx].item()
                right_edge = edges[valley_idx + 1].item()
                th = 0.5 * (left_edge + right_edge)
            else:
                # Fallbacks
                v_flat = channel.reshape(-1)
                if v_flat.var().item() < 1e-10:
                    th = float(v_flat.mean().item())  # nearly constant
                else:
                    # Try simple Otsu (coarse) with same histogram
                    total = v_flat.numel()
                    probs = counts.float() / total
                    bin_centers = 0.5 * (edges[:-1] + edges[1:])
                    w0 = torch.cumsum(probs, dim=0)
                    w1 = 1 - w0
                    mu0 = torch.cumsum(probs * bin_centers, dim=0) / (w0 + 1e-12)
                    muT = (probs * bin_centers).sum()
                    mu1 = (muT - torch.cumsum(probs * bin_centers, dim=0)) / (w1 + 1e-12)
                    between = w0 * w1 * (mu0 - mu1) ** 2
                    otsu_idx = torch.argmax(between).item()
                    th_otsu = float(bin_centers[otsu_idx].item())
                    # Blend with percentile fallback to avoid extreme cases
                    perc = float(torch.quantile(v_flat, percentile_fallback))
                    th = 0.5 * (th_otsu + perc)
            thresholds[b] = th

        # Build masks (B,H,W) then slice to prompt channels
        # (Use rescaled map for thresholding, as histogram was computed there.)
        mask_full = (rescaled >= thresholds.view(1, 1, B)).permute(2, 0, 1).long()
        return mask_full[:len(prompts)], attn_scores



    @torch.no_grad()
    def attention_map2threshold(
        self,
        attn_scores: torch.Tensor,
        prompts: List[str],
        threshold: float = 0.65,
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

        # 1) compute per‐map mean & max in a fully vectorized way
        flat = attn_scores.view(-1, B)  # (H*W, B)
        means = flat.mean(dim=0).view(1, 1, B)  # (1,1,B)
        maxs = flat.max(dim=0).values.view(1, 1, B)  # (1,1,B)
        denom = (maxs - means).clamp(min=1e-8)  # (1,1,B)

        # 2) rescale from mean→max, clamp to [0,1]
        rescaled = ((attn_scores - means) / denom).clamp(0.0, 1.0)

        # 1) direct per‐channel threshold → bool mask (H, W, B)
        bool_map = rescaled >= threshold  # True wherever score >= threshold

        # 2) permute to (B, H, W) and cast to int
        mask = bool_map.permute(2, 0, 1).long()  # (B, H, W)

        # 3) drop any background channels if you want only the prompt ones
        return mask[: len(prompts)], attn_scores



    def keep_largest_component_fill_holes(self, masks: torch.Tensor) -> torch.Tensor:
        """
        masks: (B,H,W) binary {0,1} torch.Tensor
        Returns: (B,H,W) cleaned masks (largest component, holes filled)
        """
        assert masks.dim() == 3
        out = []
        for m in masks:
            arr = (m.detach().cpu().numpy() > 0).astype(np.uint8)

            # Label connected components (4- or 8-connectivity; use structure for 8)
            structure = np.array([[1,1,1],
                                  [1,1,1],
                                  [1,1,1]], dtype=np.uint8)  # 8-connectivity
            lbl, num = ndimage.label(arr, structure=structure)
            if num == 0:
                out.append(torch.zeros_like(m))
                continue

            # Find largest component (exclude background label 0)
            sizes = ndimage.sum(arr, lbl, index=range(1, num+1))
            largest_label = 1 + int(np.argmax(sizes))
            largest = (lbl == largest_label)

            # Fill holes inside that component
            filled = ndimage.binary_fill_holes(largest)

            out.append(torch.from_numpy(filled.astype(np.uint8)).to(m.device))
        return torch.stack(out, dim=0)

    def compute_attention_with_positional_embedding(
        self,
        features_hw_c: torch.Tensor,  # (H, W, C)
        text_feats_b_c: torch.Tensor,  # (B, C)
    ) -> torch.Tensor:  # → (H, W, B)
        H, W, C = features_hw_c.shape
        B, _ = text_feats_b_c.shape

        # 1) normalize
        f = F.normalize(features_hw_c, dim=-1)  # (H,W,C)
        t = F.normalize(text_feats_b_c, dim=-1)  # (B,C)

        # 2) standard attention score
        attn = torch.einsum("hwc,bc->hwb", f, t)  # (H,W,B)

        # 3) find the argmax location for each text‐class b
        flat = attn.view(-1, B)  # (H*W, B)
        idx = flat.argmax(dim=0)  # (B,) flat indices
        ys = idx // W
        xs = idx % W
        pos_b2 = torch.stack([xs, ys], dim=-1).float().to(features_hw_c.device)  # (B,2)

        # 4) build the positional grid for all pixels
        y_coords, x_coords = torch.meshgrid(
            torch.arange(H, device=features_hw_c.device),
            torch.arange(W, device=features_hw_c.device),
            indexing="ij",
        )
        pos_hw2 = torch.stack([x_coords, y_coords], dim=-1).float()  # (H, W, 2)

        # 5) compute embeddings
        pos_emb_hw = positional_embedding(
            pos_hw2, x_max=W - 1, y_max=H - 1
        )  # (H, W, 4)
        pos_emb_b = positional_embedding(pos_b2, x_max=W - 1, y_max=H - 1)  # (B, 4)

        # 6) positional affinity score between each pixel and each class‐peak
        pos_score = torch.einsum("hwe,be->hwb", pos_emb_hw, pos_emb_b)  # (H, W, B)

        # 7) combine
        return attn + self.position_embedding_weight * pos_score

    def mIoU(
        self,
        segmentation_masks: torch.Tensor,
        masks_tensor: torch.Tensor,
        weighted: bool = False,
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

        H, W = masks_tensor[0].shape  # (height, width)
        B, H1, W1 = segmentation_masks.shape
        assert (
            H == H1 and W == W1
        ), "The shape of the predicted and ground truth masks must be the same"

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
        mode: Literal["feature_map", "attention_map"],
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

        if mode == 'attention_map':
            segmentation_masks = metrics[
                "segmentation_masks"
            ]  # Notice that the shape is (B, H, W), sparse mask
            result = self.visualize_masks(unique_category_text, segmentation_masks, rendered_image, ground_truth_image, masks_tensor)
        elif mode == 'feature_map':
            gt_feature = masks_tensor
            pred_feature = metrics["feature_map"]
            result = self.visualize_feature_map(pred_feature, gt_feature, rendered_image, ground_truth_image)


        return result


    def plot_images(self, rendered_image: torch.Tensor, 
    gt_image: torch.Tensor, 
    pca_or_seg: np.ndarray, 
    gt_pca_or_seg: np.ndarray, 
    mode: Literal["attention_map", "feature_map"], 
    unique_category_text: Union[List[str], None] = None,
    colors: Union[np.ndarray, None] = None
    ) -> torch.Tensor:

        """
        Plot the images
        """
  
        rendered_np = rendered_image.cpu().numpy()
        gt_np = gt_image
        rendered_seg = pca_or_seg
        gt_seg = gt_pca_or_seg

        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs[0, 0].imshow(rendered_np)
        axs[0, 0].set_title("Rendered Image")
        axs[0, 0].axis("off")

        axs[0, 1].imshow(gt_np)
        axs[0, 1].set_title("Ground Truth Image")
        axs[0, 1].axis("off")

        axs[1, 0].imshow(rendered_seg)
        axs[1, 0].set_title("Rendered Segmentation" if mode == "attention_map" else "Rendered Feature PCA")
        axs[1, 0].axis("off")

        axs[1, 1].imshow(gt_seg)
        axs[1, 1].set_title("Ground Truth Segmentation" if mode == "attention_map" else "Ground Truth Feature PCA")
        axs[1, 1].axis("off")

        from matplotlib.patches import Patch
        if unique_category_text is not None:
            num_classes = len(unique_category_text)

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




    def visualize_masks(self, unique_category_text: List[str], segmentation_masks: torch.Tensor, rendered_image: torch.Tensor, ground_truth_image: torch.Tensor, masks_tensor: torch.Tensor) -> torch.Tensor:

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
            segmentation_masks,  # your sparse (B,H,W)
            colors,  # RGBA colormap array
            alpha=1.0,
        )

                # GT segmentation mask (color)
        gt_seg = np.zeros((H, W, 3), dtype=np.uint8)
        for i in range(masks_tensor.shape[0]):
            mask = masks_tensor[i].cpu().numpy()
            gt_seg[mask == 1] = (colors[i][:3] * 255).astype(np.uint8)

        return self.plot_images(
            rendered_image=rendered_image,
            gt_image=ground_truth_image,
            pca_or_seg=rendered_seg,
            gt_pca_or_seg=gt_seg,
            mode="attention_map",
            unique_category_text=unique_category_text,
            colors=colors,
        )

    def visualize_feature_map(self, feature_map: torch.Tensor, gt_feature: torch.Tensor, rendered_image: torch.Tensor, ground_truth_image: torch.Tensor) -> torch.Tensor:
        """
        Visualize the feature map
        """

        # Use the same PCA fit on the concatenation of both feature maps, then transform each separately

        # Prepare both feature maps for PCA
        h1, w1, c1 = feature_map.shape
        h2, w2, c2 = gt_feature.shape
        assert c1 == c2, "Feature channel dimensions must match"

        feature_map_flat = feature_map.reshape(-1, c1)
        gt_feature_flat = gt_feature.reshape(-1, c2)

        # Fit PCA on the concatenated features
        concat_features = torch.cat([feature_map_flat, gt_feature_flat], dim=0).cpu().numpy()
        pca = PCA(n_components=3)
        pca.fit(concat_features)

        # Transform each feature map using the same PCA
        feature_pca = pca.transform(feature_map_flat.cpu().numpy()).reshape(h1, w1, 3)
        gt_feature_pca = pca.transform(gt_feature_flat.cpu().numpy()).reshape(h2, w2, 3)

        # Normalize each channel independently for both maps
        for arr in [feature_pca, gt_feature_pca]:
            for i in range(3):
                min_val = arr[..., i].min()
                max_val = arr[..., i].max()
                if max_val > min_val:
                    arr[..., i] = (arr[..., i] - min_val) / (max_val - min_val)
                else:
                    arr[..., i] = 0.0
            np.clip(arr, 0.0, 1.0, out=arr)

        return self.plot_images(
            rendered_image=rendered_image,
            gt_image=ground_truth_image,
            pca_or_seg=feature_pca,
            gt_pca_or_seg=gt_feature_pca,
            mode="feature_map",
        )



def _colorize_sparse_masks(
    sparse_masks,  # torch sparse_coo_tensor or dense Tensor, shape (B,H,W)
    colors,  # np.ndarray of shape (num_classes,4) RGBA
    alpha: float = 0.3,
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
    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)
    return canvas
