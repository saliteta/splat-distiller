
from nerfview import apply_float_colormap
import torch
from typing import Union, Literal, List, Dict
import argparse
from gsplat_ext import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive, Primitive
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from pathlib import Path
import json
from sklearn.decomposition import PCA
import viser.transforms as tf
import numpy as np
import math
from gsplat_ext import TextEncoder
from tqdm import tqdm
import imageio

# Camera intrinsics for fov=60 degrees, height=1920, width=1080
fov = 60  # degrees
height = 1080
width = 1920

# Compute focal length in pixels (vertical fov)
fov_rad = math.radians(fov)
focal_length = (height / 2) / math.tan(fov_rad / 2)

K = torch.tensor([
    [focal_length, 0, width / 2],
    [0, focal_length, height / 2],
    [0, 0, 1]
], dtype=torch.float32)



promts = ["jake the dog", "waldo", "table", "rubber yellow duck", "green apple", "green chair", "red apple", "red chair", "bag of snacks with pikachu on it", "blue elephant", "texture", "background", "porcelain", "white", "red box", "toys", "red sticks"]



def get_promts(promts):
    text_encoder = TextEncoder(model_name="SAMOpenCLIP", device="cuda")
    with torch.no_grad():
        text_features = text_encoder.encode_text(promts)
    
    text_features = torch.nn.functional.normalize(text_features, dim=1)
    return text_features


C0 = 0.28209479177387814

def RGB2SH(rgb):
    return (rgb - 0.5) / C0

def feature_to_sh(features_pca:torch.Tensor) -> torch.Tensor:
    """
    Convert features to SH coefficients
    Args:
        features_pca: (N, 3) tensor of features normalized to [0, 1]
    Returns:
        shs: (N, 3, (sh_degree + 1) ** 2) tensor of SH coefficients
    """
    sh0 = RGB2SH(features_pca)
    return sh0
    

class NeRFStudioDataset:
    def __init__(self, json_path = Path("results/camera_paths/default.json")):
        with open(json_path, "r") as f:
            self.data = json.load(f)['camera_path']
        
    def __getitem__(self, index):
        pose = tf.SE3.from_matrix(
            np.array(self.data[index]["camera_to_world"]).reshape(4, 4)
        )
        # apply the x rotation by 180 deg
        pose = tf.SE3.from_rotation_and_translation(
            pose.rotation() @ tf.SO3.from_x_radians(np.pi),
            pose.translation(),
        )

        T=pose.translation()
        R=pose.rotation().as_matrix()
        T_vec = T * 10

        pose_mat = np.eye(4, dtype=R.dtype)
        pose_mat[:3, :3] = R
        pose_mat[:3,  3] = T_vec
        pose_tensor = torch.tensor(pose_mat, dtype=torch.float32, device='cuda')

        
        return pose_tensor

    def __len__(self):
        return len(self.data)



class Renderer:
    """
    This class is used to render the scene from the camera path, and then project the features to the scene.
    The input should be camera path, feature, as well as the splats
    Another input is the output location and the mode
    """
    def __init__(self, camera_path: Path, mode: Literal["RGB", "AttentionMap", "Featuure"] = "RGB", ):
        self.camera_path = camera_path
        self.mode = "RGB" if mode == 'RGB' else "Feature"
        self.render_dataset = NeRFStudioDataset(camera_path)


    def render_selection(self, splats: Primitive):
        if isinstance(splats, GaussianPrimitive2D):
            renderer = GaussianRenderer2D(splats)
        elif isinstance(splats, BetaSplatPrimitive):
            renderer = BetaSplatRenderer(splats)
        elif isinstance(splats, GaussianPrimitive):
            renderer = GaussianRenderer(splats) 
        else:
            raise ValueError(f"Invalid splat type: {type(splats)}")
        return renderer

    
    def render_scene(self, splats: Primitive, output_location: Path = Path("results/video_outputs.mp4"), feature: Union[torch.Tensor, None] = None):
        self.renderer = self.render_selection(splats)
        if feature is not None:
            splats._feature = feature
            splats.to('cuda')
        
        for i, data in tqdm(
            enumerate(self.render_dataset),
            desc="Project the features",
            total=len(self.render_dataset),
        ):
            camtoworlds = torch.tensor(data).unsqueeze(0).cuda()
            Ks = torch.tensor(K).unsqueeze(0).cuda()

            rgbs = self.renderer.render(
                K=Ks, extrinsic=camtoworlds, width=width, height=height, mode=self.mode
            )
            # Reshape rgbs to a 1D array and plot its histogram to visualize the distribution
            import matplotlib.pyplot as plt

            rgb_flat = rgbs.detach().cpu().reshape(-1)
            # Encode the generated RGB frames into a video file.
            # Assume rgbs is a torch.Tensor of shape (H, W, C) or (C, H, W)
            import os
            import cv2
            import numpy as np

            # Create output directory if it doesn't exist
            os.makedirs(output_location.parent, exist_ok=True)

            # Prepare video writer on first frame
            if i == 0:
                img = rgbs
                if isinstance(img, torch.Tensor):
                    img = img.detach().cpu()
                    if img.dim() == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
                        img = img.permute(1, 2, 0)
                    elif img.dim() == 3 and img.shape[-1] in [1, 3]:  # (H, W, C)
                        pass
                    else:
                        raise ValueError(f"Unexpected image shape: {img.shape}")
                    img = img.clamp(0, 1)
                else:
                    raise ValueError("rgbs is not a torch.Tensor")
                h, w, c = img.shape
                video_path = output_location
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                fps = 24
                video_writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
                # Store the writer for later use
                self._feature_video_writer = video_writer
            else:
                video_writer = self._feature_video_writer

            # Convert tensor to numpy and uint8 for video
            img = rgbs
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()

                if img.dim() == 3 and img.shape[0] in [1, 3]:  # (C, H, W)
                    img = img.permute(1, 2, 0)
                elif img.dim() == 3 and img.shape[-1] in [1, 3]:  # (H, W, C)
                    pass
                else:
                    raise ValueError(f"Unexpected image shape: {img.shape}")
                img = img.clamp(0, 1)
                img_np = (img.numpy() * 255).astype(np.uint8)
            else:
                raise ValueError("rgbs is not a torch.Tensor")

            # If grayscale, convert to 3 channels for video
            if img_np.shape[2] == 1:
                img_np = np.repeat(img_np, 3, axis=2)
            # Convert RGB to BGR for OpenCV
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            video_writer.write(img_bgr)

            # On last frame, release the video writer
            if i == len(self.render_dataset) - 1:
                video_writer.release()
                del self._feature_video_writer


class Feature_selector:
    def __init__(self, mode: Literal["RGB", "AttentionMap", "Feature", "Segmentation"] = 'RGB', text_encoder: TextEncoder = TextEncoder(model_name="SAMOpenCLIP", device="cuda")):
        self.mode = mode
        self.text_encoder = text_encoder
        self.background_world = [
            "background",
            "texture",
            'object',
            "table",
            "floor",
            "wall",
            "ceiling",
            "floor",
            "wall",
            "ceiling"
            ]

    def encode_text(self, text: str):
        with torch.no_grad():
            text_features = self.text_encoder.encode_text(text)
        return text_features

    def select_feature(self, feature: torch.Tensor, text: Union[List[str], None] = None):
        """
        Return Feature and Mask
        RGB means no feature, no mask
        AttentionMap means feature is the attention map, mask is none
        Feature means feature's PCA is the feature, None
        Segmentation means feature is None, mask is the segmentation mask
        """
        if self.mode == "RGB":
            return self.rgb_preprocessing(feature)
        elif self.mode == "AttentionMap" :
            return self.attention_map_preprocessing(feature, text)
        elif self.mode == "Feature":
            return self.feature_preprocessing(feature)
        elif self.mode == "Segmentation":
            return self.segmentation_preprocessing(feature, text)

    def rgb_preprocessing(self, feature: torch.Tensor):
        return None, None
    
    def attention_map_preprocessing(self, feature: torch.Tensor, text: List[str]):
        self.text_features = self.encode_text(text)
        attention_map = torch.einsum("fc,bc->fb", feature, self.text_features)
        mean = attention_map.mean(dim=0)
        attention = attention_map.clamp(min=mean, max=attention_map.max())
        attention = (attention - mean) / (attention.max() - attention.min() + 1e-8)  # scale to roughly [-0.5, 0.5]
        attention = attention.permute(1, 0)
        attention_color = []
        for i in tqdm(range(attention.shape[0]), desc="Attention Map Preprocessing"):
            attention_color.append(apply_float_colormap(attention[i].reshape(-1, 1), colormap="turbo"))
        attention_color = torch.stack(attention_color)
        return attention_color, None
    
    def feature_preprocessing(self, feature: torch.Tensor):
        features_np = feature.cpu().numpy()
        features_np = features_np.reshape(features_np.shape[0], -1)
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_np)
        features_pca = torch.from_numpy(features_pca).float().to('cuda')
        mins   = features_pca.min(dim=0).values    # shape (3,)
        maxs   = features_pca.max(dim=0).values    # shape (3,)
        
        # 2) compute range and add eps to avoid zero-div
        ranges = maxs - mins
        eps    = 1e-8
        
        # 3) normalize into [0,1]
        features_pca = (features_pca - mins) / (ranges + eps)
        return features_pca, None

    
    def segmentation_preprocessing(self, feature: torch.Tensor, query: List[str]):
        self.text_features = self.encode_text(query+self.background_world)
        attention_map = torch.einsum("fc,bc->fb", feature, self.text_features)
        # Convert attention_map to one-hot mask of shape (h, w, b)
        # attention_map: (num_points, num_classes)
        # We assume you want a one-hot mask for each point over classes (b = num_classes)
        one_hot_mask = torch.nn.functional.one_hot(attention_map.argmax(dim=1), num_classes=attention_map.shape[1])[:, :len(query)]
        return None, one_hot_mask.permute(1, 0)


class Splat_feature_helper:
    def __init__(self, ckpt_location, method):
        self.ckpt_location = ckpt_location
        self.method = method

    def prepare_input_feature(self, mask:Union[torch.Tensor, None], features:Union[torch.Tensor, None]):
        splat:Primitive = set_splat(self.ckpt_location, self.method)
        splat.mask(mask.bool())
        return splat


def set_splat(ckpt_location, method) -> Primitive:
    if method == "2DGS":
        splat = GaussianPrimitive2D()
        splat.from_file(ckpt_location, tikhonov=1.0, feature_path=None)
    elif method == "DBS":
        splat = BetaSplatPrimitive()
        splat.from_file(ckpt_location, feature_path=None)
    elif method == "3DGS":
        splat = GaussianPrimitive()
        splat.from_file(ckpt_location, tikhonov=1.0, feature_path=None)
    else:
        raise ValueError(f"Invalid splat method: {method}")
    return splat

def rgb_rendering(ckpt_location, method, camera_path, output_path:Path):
    splat = set_splat(ckpt_location, method)
    renderer = Renderer(camera_path, mode="RGB")
    renderer.render_scene(splat, output_location = output_path)

def attention_map_rendering(ckpt_location, method, feature_path, camera_path, output_path:Path, positive_text:List[str],):
    splat = set_splat(ckpt_location, method)
    features = torch.load(feature_path)
    feature_selector = Feature_selector(mode="AttentionMap")
    attention_colors, _ = feature_selector.select_feature(features, positive_text)
    renderer = Renderer(camera_path, mode="AttentionMap")
    
    for i, attention_color in enumerate(attention_colors):
        renderer.render_scene(splat, output_location = output_path/f"{positive_text[i]}.mp4", feature=attention_color)

def feature_rendering(ckpt_location,  feature_path, method, camera_path, output_path:Path):
    splat = set_splat(ckpt_location, method)
    features = torch.load(feature_path)
    feature_selector = Feature_selector(mode="Feature")
    features, _ = feature_selector.select_feature(features)
    renderer = Renderer(camera_path, mode="Feature")
    renderer.render_scene(splat, output_location = output_path, feature=features)


def segmentation_rendering(ckpt_location,  feature_path, method, camera_path, output_path:Path, positive_text:List[str], ):
    features = torch.load(feature_path)
    feature_selector = Feature_selector(mode="Segmentation")
    _, mask = feature_selector.select_feature(features, positive_text)
    splat_feature_helper = Splat_feature_helper(ckpt_location, method)
    renderer = Renderer(camera_path, mode="RGB")
    for i in tqdm(range(mask.shape[0]), desc="Segmentation Rendering"):
        masked_splats = splat_feature_helper.prepare_input_feature(mask[i], None)
        renderer.render_scene(masked_splats, output_location = output_path/f"{positive_text[i]}.mp4")




def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_location", type=str, required=True, help="The location of the ckpt")
    parser.add_argument("--feature_path", type=str, required=True, help="The location of the feature")
    parser.add_argument("--camera_path", type=str, required=True, help="The location of the camera path")
    parser.add_argument("--output_path", type=str, required=True, help="The location of the output")
    parser.add_argument("--mode", type=str, required=True, help="The mode of the rendering")
    parser.add_argument("--method", type=str, required=True, help="The method of the rendering")
    return parser.parse_args()


def main():
    args = arg_parser()
    ckpt_location = Path(args.ckpt_location)
    feature_path = Path(args.feature_path)
    camera_path = Path(args.camera_path)
    output_path = Path(args.output_path)
    mode = args.mode
    method = args.method

    if mode == "RGB":   
        rgb_rendering(ckpt_location, method, camera_path, output_path)    
    elif mode == "Feature":
        feature_rendering(ckpt_location, feature_path, method, camera_path, output_path)    
    elif mode == "AttentionMap":
        attention_map_rendering(ckpt_location, method, feature_path, camera_path, output_path, promts)    
    elif mode == "Segmentation":
        segmentation_rendering(ckpt_location, feature_path, method, camera_path, output_path, promts)    


if __name__ == "__main__":
    main()