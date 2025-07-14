import torch
from datasets.colmap import Dataset, Parser
from utils import set_random_seed
from gsplat.distributed import cli
import argparse
from gsplat_ext import inverse_rasterization_3dgs, inverse_rasterization_2dgs
import os
from tqdm import tqdm
import torch.nn.functional as F
from analysis import compute_prototypes, fast_hdbscan_with_pos_gpu
from typing import Tuple
from primitives import GaussianPrimitive, GaussianPrimitive2D
from functools import cached_property

def points_in_frustum(pts_world: torch.Tensor,
                      cam_to_world: torch.Tensor,
                      K: torch.Tensor,
                      img_size: tuple[int,int],
                      near: float = 1e-3,
                      far: float = 1e3) -> torch.Tensor:
    """
    Args:
      pts_world: (N,3)  world‐coordinates
      cam_to_world: (4,4) camera→world homogeneous matrix
      K: (3,3)           intrinsic matrix
      img_size: (W, H)   image width and height in pixels
      near, far:         clipping planes along Z_cam
    
    Returns:
      mask: (N,) boolean, True if point is inside the frustum.
    """
    N = pts_world.shape[0]
    device = pts_world.device

    # 1) world→camera
    world_to_cam = torch.inverse(cam_to_world.to(device))         # (4,4)
    pts_h = torch.cat([pts_world, torch.ones(N,1, device=device)], 1)  # (N,4)
    pts_cam_h = (world_to_cam @ pts_h.T).T                       # (N,4)
    pts_cam = pts_cam_h[:, :3]                                   # (N,3)
    Xc, Yc, Zc = pts_cam.unbind(1)

    # 2) depth‐mask: in front and within [near,far]
    depth_mask = (Zc > near) & (Zc < far)

    # 3) project to pixel coords
    #    in homogeneous: [u·Z; v·Z; Z] = K @ [Xc; Yc; Zc]
    proj = (K.to(device) @ pts_cam.T).T                           # (N,3)
    u = proj[:,0] / proj[:,2]
    v = proj[:,1] / proj[:,2]

    # 4) image‐bounds mask
    W, H = img_size
    bounds_mask = (u >= 0) & (u < W) & (v >= 0) & (v < H)

    # 5) combine
    return depth_mask & bounds_mask



class Runner:
    """Engine for training and testing."""

    def __init__(self, args) -> None:
        set_random_seed(42)

        self.args = args
        self.device = f"cuda"


        self.set_dataset(args)
        self.set_splat(args)


    def set_dataset(self, args):
        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=args.data_dir,
            factor=args.data_factor,
            test_every=args.test_every,
        )
        trainset = Dataset(
            self.parser,
            split="train",
            load_features=True,
            feature_folder=args.feature_folder,
        )
        valset = Dataset(self.parser, split="val")

        self.trainLoader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
        self.valLoader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=False)

        self.scene_scale = self.parser.scene_scale * 1.1


    
    def set_splat(self, args):
        if args.splat_method == "2DGS":
            self.splats = GaussianPrimitive2D()
            self.splats.from_file(args.ckpt)
        elif args.splat_method == "3DGS":
            self.splats = GaussianPrimitive()
            self.splats.from_file(args.ckpt)
        else:
            raise ValueError(f"Invalid splat method: {args.splat_method}")
        
        self.splats.to(self.device)

        if args.filter:
            filter_set = Dataset(self.parser, split="train")
            filter_loader = torch.utils.data.DataLoader(filter_set, batch_size=1, shuffle=False)
            masks = self.filtering(filter_loader, self.splats.geometry["means"], torch.device(self.device))
            self.splats.geometry["means"] = self.splats.geometry["means"][masks]
            self.splats.geometry["quats"] = self.splats.geometry["quats"][masks]
            self.splats.geometry["scales"] = self.splats.geometry["scales"][masks]
            self.splats.geometry["opacities"] = self.splats.geometry["opacities"][masks]
            self.splats.color["colors"] = self.splats.color["colors"][masks]
            self.splats.save(args.ckpt.replace(".pt", "_filtered.pt"))
            
    def filtering(self, trainloader: torch.utils.data.DataLoader, means: torch.Tensor, device: torch.device, threshold: float = 0.2) -> torch.Tensor:
        """
        Filtering the splats that is not in the center of the image
        This is a simple way to filter the splats that is not in the center of the image
        Although it is very trivial, but it align with the user request for reconstructing 
        objects and fast segmentation. For example, fast 3D assets generation.

        We are not using this function for evaluation, but for real world application.


        Args:
            trainloader: the dataloader for training
            means: the means of the splats
            device: the device to use
            threshold: the threshold for filtering

        Returns:
            splat_mask: the mask for the splats
        """
        splat_weights = torch.zeros(means.shape[0], 1, device=device) # all splats are filtered
        if self.filter == False:
            return torch.ones(means.shape[0], 1) # no filtering
        else:
            for data in tqdm(trainloader, desc="Filtering splats", total=len(trainloader)):
                camtoworlds = data["camtoworld"].to(device)
                Ks = data["K"].to(device)
                height, width = data["image"].shape[1:3]
                
                splat_weights += points_in_frustum(means, camtoworlds, Ks, (width, height), near=1e-3, far=1e3).to(torch.float32)
        splat_mask = torch.topk(splat_weights, k=int(splat_weights.shape[0] * threshold), dim=0)[1]
        return splat_mask.squeeze(-1)

    def quantize(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Quantize the features to the prototypes
        We don't think this is a good idea when dealing with general tasks
        for example, lifting resnet, vit, or general features, especially visual features
        such as SHIFT features are not good for this, because they are not supposed to be quantized

        For 3D segmentation, it is also not a good idea, we recommend to use the original features
        plus our contrastive comparison to segment the splats. Since one can observe the background

        I implement this function is simply it works for metrics evaluation. During real application 
        we find that using contrastive comparison is good enough. HDBSCAN for large scale clustering 
        is too slow, and the level is hard to control.

        For now, we only use this function for metrics evaluation, not for real application.
        """
        labels, probs, pca_feat, pca_pos = fast_hdbscan_with_pos_gpu(
            embeddings=self.splat_features,
            positions=None,
            num_freqs=4,
            n_components=50,
            min_samples=10,
            use_incremental_pca=False
        )
        labels = torch.tensor(labels)
        prototypes, assigned_features = compute_prototypes(torch.tensor(labels), self.splat_features)
        return assigned_features, prototypes, labels

    @cached_property
    def inverse_rasterization(self):
        if self.args.splat_method == "2DGS":
            return inverse_rasterization_2dgs
        elif self.args.splat_method == "3DGS":
            return inverse_rasterization_3dgs
        else:
            raise ValueError(f"Invalid splat method: {self.args.splat_method}")


    @torch.no_grad()
    def distill(self, args):
        """Entry for distillation."""
        print("Running distillation...")
        means, quats, scales, opacities = self.splats.geometry["means"], self.splats.geometry["quats"], self.splats.geometry["scales"], self.splats.geometry["opacities"]
        # we can deal with 2DGS and 3DGS differently

        feature_dim = self.trainLoader.dataset[0]["features"].shape[-1]
        self.splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=self.device
        )
        self.splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=self.device
        )

        for i, data in tqdm(enumerate(self.trainLoader), desc="Distilling features", total=len(self.trainLoader)):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device) / 255.0
            height, width = pixels.shape[1:3]
            features = data["features"].to(self.device)

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
            ) = self.inverse_rasterization(
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


            self.splat_features[ids] += splat_features_per_image
            self.splat_weights[ids] += splat_weights_per_image
            del splat_features_per_image, splat_weights_per_image, ids
            torch.cuda.empty_cache()

        self.splat_features /= self.splat_weights[..., None]
        self.splat_features = torch.nan_to_num(self.splat_features, nan=0.0)

        basename, _ = os.path.splitext(args.ckpt)
        if self.args.filter:
            torch.save(self.splat_features, basename + "_filtered_features.pt")
        else:
            torch.save(self.splat_features, basename + "_features.pt")

        if self.args.quantize == "True":
            self.splat_features, self.prototypes, self.labels = self.quantize()
            torch.save(self.splat_features, basename + "_quantized_features.pt")
            torch.save(self.prototypes, basename + "_prototypes.pt")
            torch.save(self.labels, basename + "_labels.pt")



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
        default=1000,
        help="Every N images there is a test image",
    )
    parser.add_argument(
        "--feature-folder",
        type=str,
        default=None,
        help="relative Path to the feature folder",
    )
    parser.add_argument(
        "--mask-folder",
        type=str,
        default=None,
        help="Path to the mask folder",
    )
    parser.add_argument(
        "--filter",
        type=bool,
        default=False,
        help="Filter the splats that is not in the center of the image",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default="False",
        help="Quantize the features to the prototypes",
    )
    parser.add_argument(
        "--splat-method",
        type=str,
        default="3DGS",
        help="The method to use for the splatting",
        choices=["3DGS", "2DGS", "DBS"],
    )
    args = parser.parse_args()
    cli(main, args, verbose=True)
