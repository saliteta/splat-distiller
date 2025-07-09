"""
This script is used to evaluate the performance of the lifting results
It should be multi-modal evaluation, including:
- Gaussian Splatting
- 2D Gaussian Splatting
- Beta Deformable Splatting
"""

from tqdm import tqdm
from gaussian_splatting.datasets.colmap import Parser, Dataset
import os
from pathlib import Path
import argparse
from gaussian_splatting.primitives import GaussianPrimitive
from evaluator_loader import lerf_evaluator
from metrics import LERFMetrics
from gaussian_splatting.text_encoder import TextEncoder
import torch


def args_parser():
    parser = argparse.ArgumentParser(description="Evaluation script parameters")
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Path to the dataset"
    )
    parser.add_argument(
        "--label-dir", type=str, required=True, help="Path to the label directory"
    )
    parser.add_argument(
        "--result-dir", type=str, required=True, help="Path to save evaluation results"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="Path to the checkpoint"
    )
    parser.add_argument(
        "--feature-ckpt",
        type=str,
        required=False,
        help="Path to the feature checkpoint, default is the same as the ckpt but with _features.pt",
    )
    parser.add_argument(
        "--text-encoder", type=str, default="maskclip", help="text encoder to use", choices=["maskclip", "SAM2OpenCLIP", "SAMOpenCLIP"]
    )
    parser.add_argument(
        "--prototype-quantize-path", type=str, required=False, help="Path to the prototype quantize path"
    )
    parser.add_argument(
        "--rendering-mode", type=str, default="RGB+AttentionMap+VIS", help="rendering mode to use", choices=["RGB", "RGB+Feature", "RGB+Feature+Feature_PCA", "RGB+AttentionMap", "RGB+AttentionMap+VIS"]
    )
    parser.add_argument(
        "--metrics", type=str, default="attention_map", help="metrics to use", choices=["attention_map", "segmentation"]
    )
    return parser.parse_args()


def load_evaluator(args):
    # Initialize parser and dataset
    parser = Parser(
        data_dir=args.data_dir,
        factor=1,
        test_every=1,
    )

    # Create validation dataset
    valset = Dataset(
        parser,
        split="val",
        load_features=False,  # Load features for evaluation
    )
    label_dir = Path(args.label_dir)

    gaussian_primitive = GaussianPrimitive()
    gaussian_primitive.from_file(args.ckpt, args.feature_ckpt)
    if args.prototype_quantize_path is not None:
        prototypes = torch.load(args.prototype_quantize_path)
        text_encoder = TextEncoder(args.text_encoder, device=torch.device("cuda"), prototypes=prototypes)
    else:
        text_encoder = TextEncoder(args.text_encoder, device=torch.device("cuda"))

    evaluator = lerf_evaluator(gaussian_primitive, valset, label_dir, text_encoder)

    return evaluator


def main():
    args = args_parser()

    # Load data
    evaluator = load_evaluator(args)
    evaluator.eval(
        Path(args.result_dir), modes=args.rendering_mode, feature_saving_mode="pt"
    )
    metrics = LERFMetrics(
        label_folder=Path(args.label_dir), rendered_folder=Path(args.result_dir), text_encoder=args.text_encoder, enable_pca=None
    )
    metrics.compute_metrics(save_path=Path(args.result_dir), mode=args.metrics)


if __name__ == "__main__":
    main()
