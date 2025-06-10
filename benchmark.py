import os
from argparse import ArgumentParser
import json
import pandas as pd

# Lists of scenes for each dataset
lerf_ovs_scenes = [
    "ramen",
    "figurines",
    "teatime",
    "waldo_kitchen",
]


parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--output_path", default="./eval")
parser.add_argument("--lerf_ovs", type=str, help="Path to lerf_ovs dataset")
parser.add_argument("--3d_ovs", type=str, help="Path to 3d ovs dataset")
parser.add_argument(
    "--skip_feature_extraction",
    action="store_true",
    help="Skip feature extraction step",
)
parser.add_argument(
    "--skip_training",
    action="store_true",
    help="Skip training step",
)
args = parser.parse_args()

if args.lerf_ovs:
    for scene in lerf_ovs_scenes:
        source = os.path.join(args.lerf_ovs, scene)
        if not args.skip_feature_extraction:
            print(f"Extracting features for {scene}...")
            os.system(f"python feature_extractor.py -s {source}")
        if not args.skip_training:
            print(f"Running Gaussian Splatting for {scene}...")
            os.system(
                rf"python gaussian_splatting/simple_trainer.py mcmc --data-dir {source} --result_dir {args.output_path}/{scene} --disable_viewer"
            )
        print(f"Distilling features for {scene}...")
        os.system(
            rf"python gaussian_splatting/distill.py --data-dir {source} --ckpt {args.output_path}/{scene}/ckpts/ckpt_29999_rank0.pt"
        )
