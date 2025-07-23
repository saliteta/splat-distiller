from dataclasses import dataclass

import argparse
from typing import Union, Tuple

@dataclass
class DataArgs:
    dir: str
    factor: int
    test_every: int
    feature_folder: str
    mask_folder: str


@dataclass
class DistillArgs:
    method: str
    ckpt: str
    filter: bool
    quantize: bool
    tikhonov: float = 1

@dataclass
class RenderingArgs:
    dir: str
    ckpt: str
    result_type: str = "attention_map" # attention_map, feature_map, drsplat
    text_encoder: str = "maskclip" # maskclip, SAM2OpenCLIP, SAMOpenCLIP
    method: str = "3DGS" # 3DGS, 2DGS, DBS, drsplat
    feature_ckpt: Union[str, None] = None
    faiss_index_path: Union[str, None] = None # only for drsplat
    tikhonov: Union[float, None] = None
    

@dataclass
class MetricsArgs:
    threshold: float = 0.62
    threshold_mode: str = "auto" # auto, or comparison I don;t remeber
    result_folder: Union[str, None] = None
    label_folder: Union[str, None] = None


def build_parser():
    parser = argparse.ArgumentParser()

    data_args = parser.add_argument_group("Data Args")  
    data_args.add_argument("--dir", type=str, default=None, help="Path to the dataset")
    data_args.add_argument("--factor", type=int, default=1, help="Downsample factor for the dataset, default is 1")
    data_args.add_argument("--test_every", type=int, default=1000, help="Every N images there is a test image, default is 1000")
    data_args.add_argument("--feature_folder", type=str, default=None, help="relative Path to the feature folder")
    data_args.add_argument("--mask_folder", type=str, default=None, help="Path to the mask folder")
    data_args.add_argument("--ckpt", type=str, default=None, help="Path to the checkpoint")

    distill_args = parser.add_argument_group("Distill Args")
    distill_args.add_argument("--method", type=str, default="3DGS", help="The method to use for the splatting, default is 3DGS", choices=["3DGS", "2DGS", "DBS"])
    distill_args.add_argument("--tikhonov", type=float, default=None, help="Tikhonov regularization parameter")
    distill_args.add_argument("--filter", type=bool, default=False, help="Filter the splats that is not in the center of the image, default is False")
    distill_args.add_argument("--quantize", type=str, default="False", help="Quantize the features to the prototypes, default is False", choices=["True", "False"])
    
    args = parser.parse_args()
    data_args = DataArgs(dir=args.dir, factor=args.factor, test_every=args.test_every, feature_folder=args.feature_folder, mask_folder=args.mask_folder)
    distill_args = DistillArgs(method=args.method, ckpt=args.ckpt, tikhonov=args.tikhonov, filter=args.filter, quantize=args.quantize)

    return data_args, distill_args


def build_rendering_parser() -> Tuple[RenderingArgs, MetricsArgs]:
    parser = argparse.ArgumentParser()
    rendering_args = parser.add_argument_group("Rendering Args", description="contain how you load checkpoints and how you render the results")
    rendering_args.add_argument("--dir", type=str, default=None, help="Path to the dataset")
    rendering_args.add_argument("--result_type", type=str, default="attention_map", help="Result type to use, default is attention_map", choices=["attention_map", "feature_map", "drsplat"])
    rendering_args.add_argument("--ckpt", type=str, default=None, help="Path to the checkpoint")
    rendering_args.add_argument("--feature_ckpt", type=str, default=None, help="Path to the feature checkpoint, default is the same as the ckpt but with _features.pt")
    rendering_args.add_argument("--text_encoder", type=str, default="maskclip", help="Text encoder to use, default is maskclip", choices=["maskclip", "SAM2OpenCLIP", "SAMOpenCLIP"])
    rendering_args.add_argument("--faiss_index_path", type=str, default=None, help="Path to the faiss index, only for drsplat")
    rendering_args.add_argument("--method", type=str, default="3DGS", help="Splat method to use, default is 3DGS", choices=["3DGS", "2DGS", "DBS", "drsplat"])
    rendering_args.add_argument("--tikhonov", type=float, default=None, help="Tikhonov regularization parameter")


    metrics_args = parser.add_argument_group("Metrics Args", description="how we set the threshold, where we save the results and where we load the labels")
    metrics_args.add_argument("--result_folder", type=str, default=None, help="Path to the result folder")
    metrics_args.add_argument("--label_folder", type=str, default=None, help="Path to the label folder")
    metrics_args.add_argument("--threshold", type=float, default=0.62, help="Threshold for the metrics, default is 0.62")
    metrics_args.add_argument("--threshold_mode", type=str, default="auto", help="Threshold mode, default is auto", choices=["auto", "comparison"])
    args = parser.parse_args()

    metrics_args = MetricsArgs(result_folder=args.result_folder, label_folder=args.label_folder, threshold=args.threshold, threshold_mode=args.threshold_mode)
    rendering_args = RenderingArgs(dir=args.dir, tikhonov=args.tikhonov, result_type=args.result_type, ckpt=args.ckpt, feature_ckpt=args.feature_ckpt, text_encoder=args.text_encoder, faiss_index_path=args.faiss_index_path, method=args.method)

    return rendering_args, metrics_args




def args_parser():
    rendering_args, metrics_args = build_rendering_parser()
    print(rendering_args)
    print(metrics_args)

if __name__ == "__main__":
    args_parser()