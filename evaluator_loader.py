"""
    This evaluator can export several things:
    1. Load the model
    2. Load the evaluation image, and do the rendering
    3. rendered result are the following: 
        - Feature 
        - Feature PCA
        - Feature with text query
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Literal, Tuple
import torch.nn.functional as F
import torch
import numpy as np
import cv2
from tqdm import tqdm
from sklearn.decomposition import PCA
import os
from torch.utils.data import DataLoader, Dataset
from gaussian_splatting.datasets.colmap import Dataset
from gaussian_splatting.primitives import Primitive, GaussianPrimitive, GaussianPrimitive2D
from renderer import Renderer, GaussianRenderer, GaussianRenderer2D
from gaussian_splatting.text_encoder import TextEncoder
import json



BACKGROUND_WORDS = [
    "floor",
    "walls",
    "ceiling",
    "background",
    "table"
]


def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    R = optimized_camera_to_world[:, :3, :3]  # 3 x 3
    T = optimized_camera_to_world[:, :3, 3:4]  # 3 x 1
    # flip the z and y axes to align with gsplat conventions
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    # analytic matrix inverse to get world2camera matrix
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0  # homogenous
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat

class CameraDataset(Dataset):
    def __init__(self, cameras: List[dict]):
        self.cameras = cameras

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

class base_evaluator(ABC):
    def __init__(self, primitives: Primitive, dataset: Dataset, gt_paths: Path):
        super().__init__()
        assert gt_paths.exists(), f"The gt_path location does not exist: {gt_paths}"
        self.gt_paths = gt_paths
        self.primitives = primitives
        self.dataset = dataset
        self.camera_dataloader = self.load_camera()
        self.renderer = Renderer(primitives)

    @abstractmethod
    def _load_camera(self) -> Dataset:
        """
        Load the camera from the dataset, the input should be a gt_paths contain many json files
        Each json file contains the name, we should retrive the camera pose according to name
        from dataset
        """

    def load_camera(self) -> DataLoader:
        """
        Load the evaluate camera pose into a sequence of Nerfstudio Camera
        Load a list of path, camera path or gt images path
        """

        cameras = self._load_camera()
        dataloader = DataLoader(cameras, batch_size=1, shuffle=False)

        return dataloader

    @torch.no_grad()
    @abstractmethod
    def _eval(
        self, modes: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA", "RGB+AttentionMap"], camera, prompt: List[str] | None = None
    ) -> List[torch.Tensor]:
        """
        Return a List of Tensors in cpu already detached
        It should support three mode, RGB, RGB+FEATURE, RGB+FEATURE+FEATURE_PCA
        the genrated result should be put inside a list, each result should be a torch tensor

        For example the last mode we have
        [torch.shape(HW3), torch.shape(HWC), torch.shape(HW3)]
        """

    @torch.no_grad()
    def eval(
        self,
        saving_path: Path,
        modes: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA", "RGB+AttentionMap", "RGB+AttentionMap+VIS"],
        feature_saving_mode: Literal["pt", "ftz"] = "pt",
    ) -> None:
        """
        We do not consider using different texts to query an attention map
        Since it can be done using our image rendering tool, or post processing
        using 2D image metrics
        """
        print(f"==== Input mode {modes}, Processing ====")

        saving_path.mkdir(exist_ok=True)
        for mode in modes.split("+"):
            RGB_path = saving_path / mode
            RGB_path.mkdir(exist_ok=True)

        if modes == "RGB":
            for camera in tqdm(self.camera_dataloader, desc="RGB Rendering", total=len(self.camera_dataloader)):
                results: List[torch.Tensor] = self._eval(modes=modes, camera=camera)
                img = results[0]
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_name = camera["image_name"][0]
                cv2.imwrite(saving_path / "RGB" / img_name, img)

        elif modes == "RGB+Feature":

            for camera in tqdm(self.camera_dataloader, desc="RGB+Feature Rendering", total=len(self.camera_dataloader)):
                results: List[torch.Tensor] = self._eval(modes=modes, camera=camera)
                img = results[0]
                feature = results[1]
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_name = camera["image_name"][0]
                cv2.imwrite(saving_path / "RGB" / img_name, img)
                if feature_saving_mode == "pt":
                    torch.save(
                        feature,
                        saving_path / "Feature" / (img_name.split(".")[0] + ".pt"),
                    )
                elif feature_saving_mode == "ftz":
                    print("FTZ FILE DETECTED Ask Yihan Fang")
                    raise NotImplementedError
                else:
                    print(
                        f"currently only support saving feature in ftz and pt format, get: {feature_saving_mode}"
                    )
                    raise NotImplementedError

        elif modes == "RGB+Feature+Feature_PCA":

            for camera in tqdm(
                self.camera_dataloader, desc="RGB+Feature+Feature_PCA Rendering", total=len(self.camera_dataloader)
            ):
                results: List[torch.Tensor] = self._eval(modes=modes, camera=camera)
                img = results[0]
                feature = results[1]
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_name = camera["image_name"][0]
                cv2.imwrite(saving_path / "RGB" / img_name, img)
                torch.save(
                    feature, saving_path / "Feature" / (img_name.split(".")[0] + ".pt")
                )
                feature = feature.numpy()
                h, w, c = feature.shape
                feature_pca = feature.reshape(-1, feature.shape[-1])
                pca = PCA(n_components=3)
                feature_pca = pca.fit_transform(feature_pca)
                feature_pca = feature_pca.reshape(feature.shape[0], feature.shape[1], 3)
                # Normalize each channel independently
                for i in range(3):
                    min_val = feature_pca[..., i].min()
                    max_val = feature_pca[..., i].max()
                    feature_pca[..., i] = (feature_pca[..., i] - min_val) / (
                        max_val - min_val
                    )
                feature_pca = np.clip(feature_pca, 0.0, 1.0)
                feature_pca = (feature_pca * 255).astype(np.uint8)
                feature_pca = cv2.cvtColor(feature_pca, cv2.COLOR_RGB2BGR)
                cv2.imwrite(saving_path / "Feature_PCA" / img_name, feature_pca)
        
        elif modes == "RGB+AttentionMap":
            for camera in tqdm(self.camera_dataloader, desc="RGB+AttentionMap Rendering", total=len(self.camera_dataloader)):
                results: List[torch.Tensor] = self._eval(modes=modes, camera=camera)
                img = results[0]
                attention_map = results[1]
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_name = camera["image_name"][0]
                cv2.imwrite(saving_path / "RGB" / img_name, img)
                torch.save(attention_map, saving_path / "AttentionMap" / (img_name.split(".")[0] + ".pt"))
        elif modes == "RGB+AttentionMap+VIS":
            for camera in tqdm(self.camera_dataloader, desc="RGB+AttentionMap+VIS Rendering", total=len(self.camera_dataloader)):
                results: List[torch.Tensor] = self._eval(modes=modes, camera=camera)
                img = results[0]
                attention_map = results[1]
                img = img.numpy()
                img = np.clip(img, 0.0, 1.0)
                img = (img * 255).astype(np.uint8)
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                img_name = camera["image_name"][0]
                cv2.imwrite(saving_path / "RGB" / img_name, img)
                torch.save(attention_map, saving_path / "AttentionMap" / (img_name.split(".")[0] + ".pt"))
                os.makedirs(saving_path / "AttentionMap" / img_name, exist_ok=True)
                self.visualize_attention_map(attention_map, camera, saving_path / "AttentionMap" / img_name)

        else:
            print(
                f"We only support following three mode: RGB, RGB+Feature, \
                RGB+Feature+Feature_PCA, but get {mode}"
            )
            raise NotImplementedError

        print(f"=== evaluation succssful :), saved at:  {saving_path} ===")


    def visualize_attention_map(self, attention_map: torch.Tensor, img: torch.Tensor, saving_path: Path) -> None:
        """
        Visualize the attention map
        """
        pass
    
class lerf_evaluator(base_evaluator):
    def __init__(self, primitives: GaussianPrimitive, dataset: Dataset, gt_paths: Path, text_encoder: TextEncoder):
        super().__init__(primitives, dataset, gt_paths)
        if isinstance(primitives, GaussianPrimitive2D):
            self.renderer = GaussianRenderer2D(primitives)
        else:
            self.renderer = GaussianRenderer(primitives)
        self.text_encoder = text_encoder


    def _load_camera(self) -> Dataset:
        """
        Load the camera from the dataset, the input should be a gt_paths contain many json files
        Each json file contains the name, we should retrive the camera pose according to name
        from dataset
        """
        cameras = []
        files = self.gt_paths.glob("*.json")
        json_basenames = {
            Path(file).stem for file in files
        }  # Get basenames without extension
        print(f"json basenames: {json_basenames}")

        for camera in self.dataset:
            camera_basename = Path(camera["image_name"]).stem
            if camera_basename in json_basenames:
                objects = json.load(open(self.gt_paths / f"{camera_basename}.json", "r"))['objects']
                objects = [obj["category"] for obj in objects]
                camera["prompts"] = list(dict.fromkeys(objects))
                cameras.append(camera)
        return CameraDataset(cameras)

    @torch.no_grad()
    def _eval(
        self, modes: Literal["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA", "RGB+AttentionMap"], camera
    ) -> List[torch.Tensor]:
        device = "cuda"
        camtoworlds = camera["camtoworld"].to(device)
        Ks = camera["K"].to(device)
        pixels = camera["image"].to(device) / 255.0
        height, width = pixels.shape[1:3]
        results = []
        feature_tensor = None
        mode_list = modes.split("+")
        if "RGB" in mode_list:
            rgb = self.renderer.render(Ks, camtoworlds, width, height, "RGB")
            results.append(rgb)

        if "Feature" in mode_list:
            feature_tensor = self.renderer.render(
                Ks, camtoworlds, width, height, "Feature"
            )
            results.append(feature_tensor)

        if "Feature_PCA" in mode_list:
            assert feature_tensor is not None, "Feature is not available"
            feature_np = feature_tensor.numpy()
            H, W, C = feature_np.shape
            flat_feat = feature_np.reshape(-1, C)
            pca = PCA(n_components=3)
            reduced_feat = pca.fit_transform(flat_feat)
            results.append(reduced_feat)
        if "AttentionMap" in mode_list:
            prompts = [s for (s,) in camera["prompts"]] + BACKGROUND_WORDS
            text_features = self.text_encoder.encode_text(prompts)
            text_features = F.normalize(text_features, dim=0)
            attention_map = self.renderer.render(Ks, camtoworlds, width, height, "AttentionMap", text_features)
            results.append(attention_map)
        return results
    
    def visualize_attention_map(self, attention_map: torch.Tensor, camera: dict, saving_path: Path) -> None:
        """
        Visualize the attention map
        """
        prompts = [s for (s,) in camera["prompts"]]

        H, W, B = attention_map.shape
        device = attention_map.device

        # 1) compute per‐map mean & max in a fully vectorized way
        flat = attention_map.view(-1, B)                          # (H*W, B)
        means = flat.mean(dim=0).view(1, 1, B)           # (1,1,B)
        maxs  = flat.max(dim=0).values.view(1, 1, B)     # (1,1,B)
        denom = (maxs - means).clamp(min=1e-8)            # (1,1,B)

        # 2) rescale from mean→max, clamp to [0,1]
        rescaled = ((attention_map - means) / denom).clamp(0.0, 1.0)
        rescaled = rescaled.permute(2, 0, 1)

        # 3) to CPU uint8
        heat_uint8 = (rescaled * 255.0).to(torch.uint8).cpu().numpy() 

        for i in range(len(prompts)):
            heatmap = heat_uint8[i]
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_VIRIDIS)
            cv2.imwrite(str(saving_path / f"{prompts[i]}.png"), heatmap)