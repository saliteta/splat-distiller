import torch
from datasets.colmap import Dataset, Parser
from utils import set_random_seed
from gsplat.distributed import cli
import argparse
from gsplat_ext import inverse_rasterization_3dgs
import os
from tqdm import tqdm
import torch.nn.functional as F


class Runner:
    """Engine for training and testing."""

    def __init__(self, args) -> None:
        set_random_seed(42)

        self.args = args
        self.device = f"cuda"

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=args.data_dir,
            factor=args.data_factor,
            test_every=args.test_every,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            load_features=True,
        )
        self.valset = Dataset(self.parser, split="val")
        self.scene_scale = self.parser.scene_scale * 1.1
        print("Scene scale:", self.scene_scale)

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

        trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=1, shuffle=False
        )
        feature_dim = self.trainset[0]["features"].shape[-1]
        splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=device
        )
        splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=device
        )

        for i, data in tqdm(enumerate(trainloader), desc="Distilling features"):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            pixels = data["image"].to(device) / 255.0
            height, width = pixels.shape[1:3]
            features = data["features"].to(device)

            # Permute features from [B, H, W, C] to [B, C, H, W]
            features = features.permute(0, 3, 1, 2)
            # Interpolate features to match the size of pixels (height, width)
            features = torch.nn.functional.interpolate(
                features,
                size=(pixels.shape[1], pixels.shape[2]),
                mode="bilinear",
                align_corners=False,
            )
            # Permute back to [B, H, W, C] if required downstream
            features = features.permute(0, 2, 3, 1)
            features = F.normalize(features, p=2, dim=-1)

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
                viewmats=torch.linalg.inv(camtoworlds),
                Ks=Ks,
                width=width,
                height=height,
            )
            splat_features[ids] += splat_features_per_image
            splat_weights[ids] += splat_weights_per_image
            del splat_features_per_image, splat_weights_per_image, ids
            torch.cuda.empty_cache()

        splat_features /= splat_weights[..., None]
        splat_features = torch.nan_to_num(splat_features, nan=0.0)

        basename, _ = os.path.splitext(args.ckpt)
        torch.save(splat_features, basename + "_features.pt")


def main(local_rank: int, world_rank, world_size: int, args):

    runner = Runner(args)
    runner.distill(args)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to the dataset",
    )
    parser.add_argument(
        "--data-factor",
        type=int,
        default=1,
        help="Downsample factor for the dataset",
    )
    parser.add_argument(
        "--test_every",
        type=int,
        default=8,
        help="Every N images there is a test image",
    )
    args = parser.parse_args()
    cli(main, args, verbose=True)
