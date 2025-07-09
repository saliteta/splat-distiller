import os
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

lerf_ovs_scenes = [
    "ramen",
    "figurines",
    "teatime",
    "waldo_kitchen",
]


def run_lerf_ovs_evaluation(args: DictConfig):
    lerf_base_path = Path(args.lerf_ovs)
    if not lerf_base_path.exists():
        raise FileNotFoundError(
            f"Lerf OVS dataset path {lerf_base_path} does not exist."
        )

    output_path = Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if not args.skip.feature_extraction:
        for scene in lerf_base_path.iterdir():
            if scene.name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene.name} as it is not in the predefined scenes list."
                )
                continue
            print(f"Extracting features for {scene}, result will be saved in {scene / args.feature_extraction.folder} ...")
            feature_extractor_path = scene / args.feature_extraction.folder
            os.system(f"python -W ignore feature_extractor.py -s {scene} --model {args.feature_extraction.method} --ouput-dir {feature_extractor_path} --sam_ckpt_path {args.feature_extraction.sam_ckpt_path}")

    if not args.skip.training:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            result_scene = output_path / scene_name / args.training.splat_method
            if args.training.splat_method == "3DGS":
                print(f"Running Gaussian Splatting for {scene_name}...")
                os.system(
                    f"python -W ignore gaussian_splatting/simple_trainer.py default --data-dir {scene} --result_dir {result_scene} --data-factor 1 --disable_viewer --random-bkgd"
                )
            elif args.training.splat_method == "2DGS":
                print(f"Running 2D Gaussian Splatting for {scene_name}...")
                os.system(
                    f"python -W ignore gaussian_splatting/simple_trainer_2dgs.py --data-dir {scene} --result_dir {result_scene} --data-factor 1 --disable_viewer --random-bkgd"
                )
            elif args.training.splat_method == "DBS":
                print(f"Running Deformable Beta Splatting for {scene_name}...")
                os.system(
                    f"python -W ignore beta_splatting/train.py -s {scene} -m {result_scene} --random-background"
                )
            else:
                raise ValueError(f"Invalid training method: {args.training.splat_method}")

    if not args.skip.lifting:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            print(f"Lifting {scene}...")
            if args.training.splat_method == "3DGS":
                ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / "ckpt_29999_rank0.pt"
                os.system(
                    f"python -W ignore gaussian_splatting/distill.py --data-dir {scene} --ckpt {ckpt} --feature-folder {args.feature_extraction.folder} --quantize {args.distillation.quantize}"
                )
            elif args.training.splat_method == "2DGS":
                raise NotImplementedError("2DGS distillation is not implemented yet")
            elif args.training.splat_method == "DBS":
                raise NotImplementedError("DBS distillation is not implemented yet")
            else:
                raise ValueError(f"Invalid training method: {args.training.splat_method}")

    if not args.skip.evaluation:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            result_scene = output_path / scene_name / args.training.splat_method
            label_path = lerf_base_path / "label" / scene_name
            ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / "ckpt_29999_rank0.pt"
            if args.distillation.quantize:
                feature_ckpt = ckpt.parent / (ckpt.stem + "_quantized_features.pt")
            else:
                feature_ckpt = None
            print(f"Evaluating {scene_name}...")
            os.system(
                f"python -W ignore eval.py --data-dir {scene} --result-dir {result_scene} --label-dir {label_path} \
                    --ckpt {ckpt} --text-encoder {args.feature_extraction.method} --feature-ckpt {feature_ckpt} \
                    --rendering-mode {args.evaluation.rendering_mode} --metrics {args.evaluation.metrics}"
            )


@hydra.main(config_path="config", config_name="for_metrics.yaml")
def main(args: DictConfig):
    run_lerf_ovs_evaluation(args)


if __name__ == "__main__":
    main()



# python benchmark.py --lerf_ovs ../lerf_ovs --output_path results --text_encoder SAM2OpenCLIP --skip_training --skip_lifting --skip_evaluation