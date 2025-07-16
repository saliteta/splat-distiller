import os
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra

lerf_ovs_scenes = [
    "figurines",
    "ramen",
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
                    f"python -W ignore gaussian_splatting/simple_trainer.py default --data-dir {scene} --result_dir {result_scene} \
                        --data-factor 1 --disable_viewer --random-bkgd --max-steps {args.training.max_steps} --save-steps {args.training.save_steps} --eval-steps {args.training.max_steps+10}"
                )
            elif args.training.splat_method == "2DGS":
                print(f"Running 2D Gaussian Splatting for {scene_name}...")
                os.system(
                    f"python -W ignore gaussian_splatting/simple_trainer_2dgs.py --data-dir {scene} --result_dir {result_scene} \
                        --data-factor 1 --disable_viewer --random-bkgd --test_every 1000 --max-steps {args.training.max_steps} --save-steps {args.training.save_steps}"
                )
            elif args.training.splat_method == "DBS":
                print(f"Running Deformable Beta Splatting for {scene_name}...")
                os.system(
                    f"python -W ignore beta_splatting/train.py -s {scene} -m {result_scene} --iterations {args.training.max_steps}"
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
                ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / f"ckpt_{args.training.max_steps-1}_rank0.pt"
                os.system(
                    f"python -W ignore distill.py --data-dir {scene} --ckpt {ckpt} --feature-folder {args.feature_extraction.folder} \
                        --quantize {args.distillation.quantize} --splat-method {args.training.splat_method}"
                )
            elif args.training.splat_method == "2DGS":
                ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / f"ckpt_{args.training.max_steps-1}.pt"
                os.system(
                    f"python -W ignore distill.py --data-dir {scene} --ckpt {ckpt} --feature-folder {args.feature_extraction.folder} \
                        --quantize {args.distillation.quantize} --splat-method {args.training.splat_method}"
                )
            elif args.training.splat_method == "DBS":
                ckpt = output_path / scene_name / args.training.splat_method / "point_cloud" / f"iteration_{args.training.max_steps}/point_cloud.ply"
                os.system(
                    f"python -W ignore distill.py --data-dir {scene} --ckpt {ckpt} --feature-folder {args.feature_extraction.folder} \
                        --quantize {args.distillation.quantize} --splat-method {args.training.splat_method} --filter {args.distillation.filter}"
                )
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
            label_path = lerf_base_path / "label" / scene_name
            if args.extension == "drsplat":
                result_scene = output_path  / args.training.splat_method / scene_name
                ckpt = output_path  / args.training.splat_method / scene_name / 'drsplat_features_1_pq_openclip_topk10_weight_128' / f"chkpnt0.pth"
                feature_ckpt = ckpt.parent / (ckpt.stem + "_features.pt")
                os.system(
                    f"python -W ignore eval.py --data-dir {scene} --result-dir {result_scene} --label-dir {label_path} \
                        --ckpt {ckpt} --text-encoder {args.feature_extraction.method} --feature-ckpt {feature_ckpt} \
                        --rendering-mode {args.evaluation.rendering_mode} --metrics {args.evaluation.metrics} --faiss-index-path {args.faiss_index_path}"
                )
            elif args.extension == "None":
                result_scene = output_path / scene_name/ args.training.splat_method 
                if args.training.splat_method == "2DGS":
                    ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / f"ckpt_{args.training.max_steps-1}.pt"
                elif args.training.splat_method == "DBS":
                    if args.distillation.filter:
                        ckpt = output_path / scene_name / args.training.splat_method / "point_cloud" / f"iteration_{args.training.max_steps}/point_cloud_filtered.pt"
                    else:
                        ckpt = output_path / scene_name / args.training.splat_method / "point_cloud" / f"iteration_{args.training.max_steps}/point_cloud.ply"
                else:
                    ckpt = output_path / scene_name / args.training.splat_method / "ckpts" / f"ckpt_{args.training.max_steps-1}_rank0.pt"
                if args.distillation.quantize:
                    feature_ckpt = ckpt.parent / (ckpt.stem + "_quantized_features.pt")
                else:
                    feature_ckpt = ckpt.parent / (ckpt.stem + "_features.pt")
                os.system(
                f"python -W ignore eval.py --data-dir {scene} --result-dir {result_scene} --label-dir {label_path} \
                    --ckpt {ckpt} --text-encoder {args.feature_extraction.method} --feature-ckpt {feature_ckpt} \
                    --rendering-mode {args.evaluation.rendering_mode} --metrics {args.evaluation.metrics} --splat-method {args.training.splat_method}"
            )
            else:
                raise ValueError(f"Invalid extension: {args.extension}")
            print(f"Evaluating {scene_name}...")



@hydra.main(config_path="config", config_name="for_metrics.yaml")
def main(args: DictConfig):
    run_lerf_ovs_evaluation(args)


if __name__ == "__main__":
    main()


