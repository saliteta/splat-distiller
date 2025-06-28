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

    evaluator = lerf_evaluator(gaussian_primitive, valset, label_dir)

    return evaluator


def main():
    args = args_parser()

    # Load data
    evaluator = load_evaluator(args)
    evaluator.eval(
        Path(args.result_dir), modes="RGB+Feature+Feature_PCA", feature_saving_mode="pt"
    )
    if args.text_encoder == "SAM2OpenCLIP":
        enable_pca = 256
    else:
        enable_pca = None
        
    metrics = LERFMetrics(
        label_folder=Path(args.label_dir), rendered_folder=Path(args.result_dir), text_encoder=args.text_encoder, enable_pca=enable_pca
    )
    metrics.compute_metrics(save_path=Path(args.result_dir))


if __name__ == "__main__":
    main()
