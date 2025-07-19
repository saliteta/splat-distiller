from nerfview import apply_float_colormap
import torch
from gsplat_ext import Dataset, Parser
from gaussian_splatting.utils import set_random_seed
from gsplat.distributed import cli
import argparse
import os
from tqdm import tqdm
import torch.nn.functional as F
from gaussian_splatting.analysis import compute_prototypes, fast_hdbscan_with_pos_gpu
from typing import Tuple, Union, Dict
from gsplat_ext import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from gsplat_ext import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from pathlib import Path
from PIL import Image
from application_beta.semantic_query.mask_label_registration import mask_register

beta_filter_args = {
    "beta_filter_larde": 3.0,
}


class Runner:
    """Engine for training and testing."""

    def __init__(self, args) -> None:
        set_random_seed(42)

        self.args = args
        self.device = f"cuda"

        self.set_dataset(args)
        self.set_splat(args)  # set the splat primitive and renderer

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

        self.trainLoader = torch.utils.data.DataLoader(
            trainset, batch_size=1, shuffle=False
        )
        self.valLoader = torch.utils.data.DataLoader(
            valset, batch_size=1, shuffle=False
        )

        self.scene_scale = self.parser.scene_scale * 1.1

    def set_splat(self, args):
        if args.splat_method == "2DGS":
            self.splats = GaussianPrimitive2D()
            self.splats.from_file(args.ckpt)
            self.renderer = GaussianRenderer2D(self.splats)
        elif args.splat_method == "3DGS":
            self.splats = GaussianPrimitive()
            self.splats.from_file(args.ckpt)
            self.renderer = GaussianRenderer(self.splats)
        elif args.splat_method == "DBS":
            self.splats = BetaSplatPrimitive()
            self.splats.from_file(args.ckpt)
            self.renderer = BetaSplatRenderer(self.splats)
        else:
            raise ValueError(f"Invalid splat method: {args.splat_method}")

        self.splats.to(self.device)

        if args.filter:
            filter_set = Dataset(self.parser, split="train")
            filter_loader = torch.utils.data.DataLoader(
                filter_set, batch_size=1, shuffle=False
            )
            masks = self.splats.filtering(
                filter_loader,
                self.splats.geometry["means"],
                torch.device(self.device),
                threshold=0.8,
                args=beta_filter_args if args.splat_method == "DBS" else None,
            )
            ckpt_path = Path(args.ckpt)
            self.splats.save(ckpt_path.parent / (ckpt_path.stem + "_filtered.pt"))

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
            use_incremental_pca=False,
        )
        labels = torch.tensor(labels)
        prototypes, assigned_features = compute_prototypes(
            torch.tensor(labels), self.splat_features
        )
        return assigned_features, prototypes, labels

    @torch.no_grad()
    def distill(self, args):
        """Entry for distillation."""
        print("Running distillation...")
        means = self.splats.geometry["means"]
        # we can deal with 2DGS and 3DGS differently

        feature_dim = self.trainLoader.dataset[0]["features"].shape[-1]
        self.splat_features = torch.zeros(
            (means.shape[0], feature_dim), dtype=torch.float32, device=self.device
        )
        self.splat_weights = torch.zeros(
            (means.shape[0]), dtype=torch.float32, device=self.device
        )

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Distilling features",
            total=len(self.trainLoader),
        ):
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
            ) = self.renderer.inverse_render(
                K=Ks,
                extrinsic=camtoworlds,
                width=width,
                height=height,
                features=features,
            )

            self.splat_features[ids] += splat_features_per_image
            self.splat_weights[ids] += splat_weights_per_image
            del splat_features_per_image, splat_weights_per_image, ids, features
            torch.cuda.empty_cache()

        self.splat_features /= self.splat_weights[..., None]
        self.splat_features = torch.nan_to_num(self.splat_features, nan=0.0)

        del self.splat_weights
        torch.cuda.empty_cache()

        self.basename, _ = os.path.splitext(args.ckpt)
        if self.args.filter:
            torch.save(self.splat_features, self.basename + "_filtered_features.pt")
        else:
            torch.save(self.splat_features, self.basename + "_features.pt")
        
        self.splat_features = self.splat_features.cpu()

        if self.args.quantize == "True":
            print("Quantize is set to True, Quantizing")
            _, _, self.labels = self.quantize()
            torch.save(self.labels, self.basename + "_labels.pt")
            if 'SAMOpenCLIP' in args.feature_folder:
                print("Detecting Using SAMOpenCLIP, apply refinement ...")
                image_labels: Dict[str, torch.Tensor] = self.label_projection(self.labels)
                del self.splats
                torch.cuda.empty_cache()
                refined = self.mask_level_refinement(args, image_labels)
                torch.save(refined, self.basename + "_refined_features.pt")


    
    def mask_level_refinement(self, args: argparse.Namespace, image_labels: Dict[str, torch.Tensor]):
        mask_folder = Path(args.data_dir) / args.feature_folder 
        # Notice that we only support SAMOpenCLIP for now, it can be further implemented to mask related
        if "SAM" not in str(args.feature_folder):
            print("we currently only support SAM OpenCLIP for refinement, theoratically it should work for all mask related results")
            raise NotImplementedError
        self.mask_registrater = mask_register(image_labels, mask_folder)
        self.mask_registrater.reg()
        refined = self.mask_registrater.GaussianFeatureRefinement(self.splat_features.cuda(), self.labels)
        return refined

    @torch.no_grad()
    def label_projection(
        self, labels: torch.Tensor, debugg_path: Union[Path, None] = None
    ):
        """
        The stratgy is the following, we can assign the label as a one hot vector, or other orthogonal vector,
        and then we project those orthogonal labels from high dimension to camera position, and then determine
        pixel level correspondance.
        That is at position (image_number i, height x ,width y) -> cluster c. If we have so called masked level
        level correspondance, we then have a mask to Gaussian Cluster Correspondance, therefore, we can somehow
        keep the multiview difference for each Gaussian

        We try not to save it on disks
        """
        print("beta version will try to establish mask to Gaussian Cluster relation")
        image_labels = {}

        # labels: (n,) input, where n is the number of elements
        labels = labels + 1  # Shift -1 to 0, 0 to 1, etc.
        num_classes = (
            int(labels.max().item()) + 1
        )  # Since -1 is now 0, this covers all classes including background
        one_hot = F.one_hot(labels.long(), num_classes=num_classes).float()
        #### Then we start to do the projection
        #### We over write the feature in this way, and trigger feature map rendering
        self.splats._feature = one_hot.cuda()
        store_folder = Path(self.basename).parent / "labels"
        store_folder.mkdir(exist_ok=True)

        for i, data in tqdm(
            enumerate(self.trainLoader),
            desc="Project the Cluster Labels",
            total=len(self.trainLoader),
        ):
            camtoworlds = data["camtoworld"].to(self.device)
            Ks = data["K"].to(self.device)
            pixels = data["image"].to(self.device)
            height, width = pixels.shape[1:3]
            image_name = Path(data["image_name"][0])

            projected_labels = self.renderer.render(
                K=Ks, extrinsic=camtoworlds, width=width, height=height, mode="Feature"
            )
            assert (
                projected_labels != None
            ), "render has some issue, expect labels as shape of H,W,m shape images"
            # Labels should be a (H,W,N) tensor, and we do argmax to get the cluster it belongs to
            label_dense = projected_labels.argmax(dim=-1)

            image_labels[image_name.stem] = label_dense.cpu()


            # Save the color-mapped image using PIL
            if debugg_path != None:
                assert isinstance(debugg_path, Path)
                map = label_dense / num_classes
                images_for_debugging = apply_float_colormap(map.reshape(-1, 1))
                images_for_debugging = images_for_debugging.reshape(height, width, 3)
                save_path = debugg_path / (image_name.stem + "jpg")
                # images_for_debugging is assumed to be a numpy array or torch tensor in HWC, float [0,1] or [0,255]
                img_np = images_for_debugging.cpu().numpy()
                img_pil = Image.fromarray(img_np)
                img_pil.save(save_path)
        
        return image_labels

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
