import os
from argparse import ArgumentParser
from pathlib import Path

lerf_ovs_scenes = [
    "ramen",
    "figurines",
    "teatime",
    "waldo_kitchen",
]


def run_lerf_ovs_evaluation(args):
    lerf_base_path = Path(args.lerf_ovs)
    if not lerf_base_path.exists():
        raise FileNotFoundError(
            f"Lerf OVS dataset path {lerf_base_path} does not exist."
        )

    output_path = Path(args.output_path)

    if not args.skip_feature_extraction:
        for scene in lerf_base_path.iterdir():
            print(f"Extracting features for {scene}...")
            os.system(f"python feature_extractor.py -s {scene}")

    if not args.skip_training:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            result_scene = output_path / scene_name
            if args.splat_method == "3DGS":
                print(f"Running Gaussian Splatting for {scene_name}...")
                os.system(
                    f"python gaussian_splatting/simple_trainer.py default --data-dir {scene} --result_dir {result_scene} --data-factor 1 --disable_viewer --random-bkgd"
                )
            elif args.splat_method == "2DGS":
                print(f"Running 2D Gaussian Splatting for {scene_name}...")
                os.system(
                    f"python gaussian_splatting/simple_trainer_2dgs.py --data-dir {scene} --result_dir {result_scene} --data-factor 1 --disable_viewer --random-bkgd"
                )
            elif args.splat_method == "DBS":
                print(f"Running Deformable Beta Splatting for {scene_name}...")
                os.system(
                    f"python beta_splatting/train.py -s {scene} -m {result_scene} --random-background"
                )
            else:
                raise ValueError(f"Invalid training method: {args.splat_method}")

    if not args.skip_lifting:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            print(f"Lifting {scene}...")
            if args.splat_method == "3DGS":
                ckpt = output_path / scene_name / "ckpts" / "ckpt_29999_rank0.pt"
                os.system(
                    f"python gaussian_splatting/distill.py --data-dir {scene} --ckpt {ckpt}"
                )
            elif args.splat_method == "2DGS":
                raise NotImplementedError("2DGS distillation is not implemented yet")
            elif args.splat_method == "DBS":
                raise NotImplementedError("DBS distillation is not implemented yet")
            else:
                raise ValueError(f"Invalid training method: {args.splat_method}")

    if not args.skip_evaluation:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            result_scene = output_path / scene_name
            label_path = lerf_base_path / "label" / scene_name
            ckpt = output_path / scene_name / "ckpts" / "ckpt_29999_rank0.pt"
            print(f"Evaluating {scene_name}...")
            os.system(
                f"python eval.py --data-dir {scene} --result-dir {result_scene} --label-dir {label_path} --ckpt {ckpt}"
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluation script parameters")
    parser.add_argument("--output_path", default="eval")
    parser.add_argument("--lerf_ovs", type=str, help="Path to lerf_ovs dataset")
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

    parser.add_argument(
        "--splat_method",
        default="3DGS",
        help="splat method to use, can be choose from Gaussian Splatting, 2DGS, beta deformable splatting",
        choices=["3DGS", "2DGS", "DBS"],
    )

    parser.add_argument(
        "--skip_lifting",
        action="store_true",
        help="Skip lifting step",
    )

    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation step",
    )

    args = parser.parse_args()

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.lerf_ovs:
        print("Running evaluation for lerf_ovs dataset...")
        run_lerf_ovs_evaluation(args)
