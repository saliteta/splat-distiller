import os
from argparse import ArgumentParser
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
import hydra
from typing import Union

lerf_ovs_scenes = [
    "figurines",
    "ramen",
    "teatime",
    "waldo_kitchen",
]


def ckpt_path(scene_name: str, output_path: Path, method: str, extension: str, max_steps: Union[int, None] = None):

    if extension == "INRIA":
        return output_path / method / scene_name / 'point_cloud/iteration_30000/scene_point_cloud.ply'
    else:
        base_path = output_path / scene_name / method
        if max_steps is None:
            raise ValueError("max_steps is required for non-INRIA scenes")
        if method == "3DGS":
            return base_path / "ckpts" / f"ckpt_{max_steps-1}_rank0.pt"
        elif method == "2DGS":
            return base_path / "ckpts" / f"ckpt_{max_steps-1}.pt"
        elif method == "DBS":
            return base_path / "point_cloud" / f"iteration_{max_steps}/point_cloud.ply"
        else:
            raise ValueError(f"Invalid method: {method}")

        


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
            print(
                f"Extracting features for {scene}, result will be saved in {scene / args.feature_extraction.folder} ..."
            )
            feature_extractor_path = scene / args.feature_extraction.folder
            os.system(
                f"python -W ignore feature_extractor.py -s {scene} --model {args.feature_extraction.method} --ouput-dir {feature_extractor_path} --sam_ckpt_path {args.feature_extraction.sam_ckpt_path}"
            )

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
                raise ValueError(
                    f"Invalid training method: {args.training.splat_method}"
                )

    if not args.skip.lifting:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            print(f"Lifting {scene}...")
            if args.extension == "INRIA":
                ckpt = ckpt_path(scene_name, output_path, args.training.splat_method, args.extension, None)
            else:
                ckpt = ckpt_path(scene_name, output_path, args.training.splat_method, args.extension, args.training.max_steps)
            

            ### We are trying to use tikhonov regularization to distill the model
            os.system(
                f"python -W ignore distill.py --dir {scene} --ckpt {ckpt} --feature_folder {args.feature_extraction.folder} \
                    --quantize {args.distillation.quantize} --method {args.training.splat_method} --tikhonov {args.distillation.tikhonov}"
            )

    if not args.skip.evaluation:
        for scene in lerf_base_path.iterdir():
            scene_name = scene.name
            if scene_name not in lerf_ovs_scenes:
                print(
                    f"Skipping {scene_name} as it is not in the predefined scenes list."
                )
                continue
            if args.extension == "INRIA":
                ckpt = ckpt_path(scene_name, output_path, args.training.splat_method, args.extension, None)
            else:
                ckpt = ckpt_path(scene_name, output_path, args.training.splat_method, args.extension, args.training.max_steps)
            
            label_path = lerf_base_path / "label" / scene_name
            if args.extension == "drsplat":
                result_scene = output_path / args.training.splat_method / scene_name
                ckpt = (
                    output_path
                    / args.training.splat_method
                    / scene_name
                    / "drsplat_features_1_pq_openclip_topk10_weight_128"
                    / f"chkpnt0.pth"
                )
                feature_ckpt = ckpt.parent / (ckpt.stem + "_features.pt")
                os.system(
                    f"python -W ignore eval.py --dir {scene} --result_folder {result_scene} --label_folder {label_path} \
                        --ckpt {ckpt} --text-encoder {args.feature_extraction.method} --feature-ckpt {feature_ckpt} \
                        --result_type {args.evaluation.result_type} --method {args.training.splat_method} --faiss-index-path {args.faiss_index_path}"
                )
            elif args.extension == "None" or args.extension == "INRIA":
                result_scene = output_path / scene_name / args.training.splat_method
                if args.extension == "INRIA":
                    result_scene = output_path / args.training.splat_method / scene_name


                

                if args.distillation.quantize and 'SAMOpenCLIP' not in args.feature_extraction.method:
                    feature_ckpt = ckpt.parent / (ckpt.stem + "_quantized_features.pt")
                elif args.distillation.quantize and 'SAMOpenCLIP' in args.feature_extraction.method:
                    feature_ckpt = ckpt.parent / (ckpt.stem + "_refined_features.pt")
                else:
                    feature_ckpt = ckpt.parent / (ckpt.stem + "_features.pt")
                os.system(
                    f"python -W ignore eval.py --dir {scene} --result_folder {result_scene} --label_folder {label_path} \
                    --ckpt {ckpt} --text_encoder {args.feature_extraction.method} --feature_ckpt {feature_ckpt} \
                    --result_type {args.evaluation.metrics} --method {args.training.splat_method}"
                )
            else:
                raise ValueError(f"Invalid extension: {args.extension}")
            print(f"Evaluating {scene_name}...")


@hydra.main(config_path="config", config_name="for_inria.yaml")
def main(args: DictConfig):
    run_lerf_ovs_evaluation(args)


if __name__ == "__main__":
    main()
