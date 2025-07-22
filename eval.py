"""
This script is used to evaluate the performance of the lifting results
It should be multi-modal evaluation, including:
- Gaussian Splatting
- 2D Gaussian Splatting
- Beta Deformable Splatting
"""

from gsplat_ext import Parser, Dataset
from pathlib import Path
from gsplat_ext import (
    GaussianPrimitive,
    DrSplatPrimitive,
    GaussianPrimitive2D,
    BetaSplatPrimitive,
)
from evaluator_loader import lerf_evaluator
from metrics import LERFMetrics
from gsplat_ext import TextEncoder
import torch
from argparser import build_rendering_parser, RenderingArgs, MetricsArgs





def load_evaluator(rendering_args: RenderingArgs, metrics_args: MetricsArgs):
    # Initialize parser and dataset
    parser = Parser(
        data_dir=rendering_args.dir,
        factor=1,
        test_every=1,
    )

    # Create validation dataset
    valset = Dataset(
        parser,
        split="val",
        load_features=False,  # Load features for evaluation
    )
    label_dir = Path(metrics_args.label_folder)

    if rendering_args.faiss_index_path is not None:
        if rendering_args.method == "drsplat":
            gaussian_primitive = DrSplatPrimitive()
            gaussian_primitive.from_file(rendering_args.ckpt, rendering_args.faiss_index_path)
        else:
            raise ValueError(f"Invalid splat method: {rendering_args.method}")
    elif rendering_args.method == "3DGS":
        gaussian_primitive = GaussianPrimitive()
        gaussian_primitive.from_file(rendering_args.ckpt, rendering_args.feature_ckpt, tikhonov=rendering_args.tikhonov)
    elif rendering_args.method == "2DGS":
        gaussian_primitive = GaussianPrimitive2D()
        gaussian_primitive.from_file(rendering_args.ckpt, rendering_args.feature_ckpt, tikhonov=rendering_args.tikhonov)
    elif rendering_args.method == "DBS":
        gaussian_primitive = BetaSplatPrimitive()
        gaussian_primitive.from_file(rendering_args.ckpt, rendering_args.feature_ckpt)

    gaussian_primitive.to(torch.device("cuda"))

    text_encoder = TextEncoder(rendering_args.text_encoder, device=torch.device("cuda"))

    evaluator = lerf_evaluator(gaussian_primitive, valset, label_dir, text_encoder)

    return evaluator


def main():
    rendering_args, metrics_args = build_rendering_parser()
    # Load data
    if rendering_args.result_type == "feature_map":
        modes = "RGB+Feature+Feature_PCA"
    elif rendering_args.result_type == "attention_map":
        modes = "RGB+AttentionMap+VIS"
    elif rendering_args.method == "drsplat":
        modes = "RGB+AttentionMap+VIS"
    else:
        raise ValueError(f"Invalid metrics: {rendering_args.result_type}")
    evaluator = load_evaluator(rendering_args, metrics_args)
    evaluator.eval(Path(metrics_args.result_folder), modes=modes, feature_saving_mode="pt")
    metrics = LERFMetrics(
        label_folder=Path(metrics_args.label_folder),
        rendered_folder=Path(metrics_args.result_folder),
        text_encoder=rendering_args.text_encoder,
        enable_pca=None,
    )
    metrics.compute_metrics(save_path=Path(metrics_args.result_folder), mode=rendering_args.result_type)


if __name__ == "__main__":
    main()
