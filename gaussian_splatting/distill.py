import torch
from datasets.colmap import Dataset, Parser
import argparse
from gsplat_ext import inverse_rasterization_3dgs
import os
from tqdm import tqdm
import torch.nn.functional as F

def get_viewmat(optimized_camera_to_world):
    """
    function that converts c2w to gsplat world2camera matrix, using compile for some speed
    """
    optimized_camera_to_world = optimized_camera_to_world.unsqueeze(0)
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


class Runner:
    """Engine for training and testing."""

    def __init__(self, args) -> None:

        self.device = f"cuda"

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=args.data_dir,
            factor=1,
        )
        self.dataset = Dataset(
            self.parser,
            split="all",
            load_features=True,
        )

    @torch.no_grad()
    def distill(self, args):
        """Entry for distillation."""
        print("Running distillation...")
        device = self.device
        self.splat = torch.load(args.ckpt, map_location=device, weights_only=True)[
            "splats"
        ]
        means, quats, scales, opacities = (
            self.splat["means"],
            F.normalize(self.splat["quats"], p=2, dim=-1),
            torch.exp(self.splat["scales"]),
            torch.sigmoid(self.splat["opacities"]),
        )

        
        feature_dim = self.dataset[0]["features"].shape[-1]
        splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=device
        )
        splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=device
        )

        for i, data in tqdm(enumerate(self.dataset), desc="Distilling features"):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device).unsqueeze(0)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[0:2]
            
            features = data["features"].to(device).permute(2,0,1).unsqueeze(0)
            
            # Interpolate features to match the size of pixels (height, width)
            
            features = torch.nn.functional.interpolate(
                features,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            ).squeeze().permute(2,0,1)
            
            features = F.normalize(features.reshape(-1, features.shape[-1]), dim=1).reshape(height,width,-1)
            # Permute back to [B, H, W, C] if required downstream
            features = features.unsqueeze(0)

            (
                splat_features_per_image,
                splat_weights_per_image,
                ids,
            ) = inverse_rasterization_3dgs(
                means=means,
                quats=quats,
                scales=scales,
                opacities=opacities,
                input_image=features,
                viewmats= get_viewmat(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
            )
            splat_features[ids] += splat_features_per_image
            splat_weights[ids] += splat_weights_per_image
            del splat_features_per_image, splat_weights_per_image, ids
            torch.cuda.empty_cache()

        splat_features /= splat_weights[..., None]
        print(splat_features.shape)
        splat_features = torch.nan_to_num(splat_features, nan=0.0)

        basename, _ = os.path.splitext(args.ckpt)
        torch.save(splat_features, basename + "_features.pt")
        print(f"successfully saved to -> {basename}_features.pt")



def parser():  
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Distill features from Gaussian Splatting")
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to the checkpoint file containing the Gaussian splats.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to the dataset directory.",
    )
    return parser.parse_args()

def main():
    args = parser()
    runner = Runner(args)
    runner.distill(args)

if __name__ == "__main__":
    main()



