import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from pre_processing import (
    OpenCLIPNetwork,
    OpenCLIPNetworkConfig,
    sam_model_registry,
    SamAutomaticMaskGenerator,
    create,
)
import cv2

"""
    This script is used to extract features from images using a specified model.
    It supports different models, we got two major families of models:
    - FeatUp Models: "dino16", "dinov2", "clip", "maskclip", "vit", "resnet50"
    - SAM2OpenCLIP: "SAM2OpenCLIP", this can be modified according to the open clip model you want to use.
"""
# Supported models
SUPPORTED_MODELS = [
    "dino16",
    "dinov2",
    "clip",
    "maskclip",
    "vit",
    "resnet50",
    "SAMOpenCLIP",
]


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        args: Parsed arguments with source folder and model name.
    """
    parser = argparse.ArgumentParser(
        description="Process images to generate RGBF arrays using specified foundation model."
    )

    parser.add_argument(
        "--source_path",
        "-s",
        type=str,
        default="data",
        help="Path to the image input folder.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="maskclip",
        choices=SUPPORTED_MODELS,
        help=f"Select the 2D foundation model from the list: {', '.join(SUPPORTED_MODELS)}.",
    )

    parser.add_argument(
        "--ouput-dir",
        type=str,
        default="features",
        help="Relative Path to the feature output folder.",
    )
    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")
    parser.add_argument(
        "--sam_ckpt_path",
        type=str,
        default="sam_vit_h_4b8939.pth",
        help="path to the sam checkpoint",
        required=False,
    )
    return parser.parse_args()


def load_upsampler(model_name, device):
    """
    Load the specified upsampler model.

    Args:
        model_name (str): Name of the model to load.
        device (torch.device): Device to load the model on.

    Returns:
        torch.nn.Module: Loaded upsampler model.
    """
    # try:
    upsampler = torch.hub.load("mhamilton723/FeatUp", model_name, use_norm=False).to(
        device
    )
    upsampler.eval()  # Set model to evaluation mode
    print(f"Successfully loaded model '{model_name}'.")
    return upsampler


def main_featup(data_dir: str, features_output_dir: str, model_name: str, device: str):

    # Define transformations
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # Load the upsampler model
    upsampler = load_upsampler(model_name, device)

    # Create output directory if it doesn't exist
    os.makedirs(features_output_dir, exist_ok=True)

    # List and sort image files
    all_files = sorted(os.listdir(os.path.join(data_dir, "images")))
    supported_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_files = [
        f for f in all_files if any(f.lower().endswith(ext) for ext in supported_ext)
    ]

    if len(image_files) == 0:
        print(
            f"No supported image files found in '{data_dir}'. Supported extensions: {', '.join(supported_ext)}."
        )
        exit(1)

    # Process each image in the directory
    for filename in tqdm(image_files, desc="Extracting features", unit="img"):
        file_path = os.path.join(data_dir, "images", filename)

        # (Optional) if you still want to skip non-images despite filtering earlier
        if not any(filename.lower().endswith(ext) for ext in supported_ext):
            tqdm.write(f"Skipping non-image file: {filename}")
            continue

        image = Image.open(file_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        # Generate high-resolution features
        with torch.no_grad():
            hr_feats = upsampler(image_tensor)

        # Permute to (H, W, F)
        hr_feats = hr_feats.squeeze(0).permute(1, 2, 0)

        # Save as a .pt
        base_name, _ = os.path.splitext(filename)
        out_path = os.path.join(features_output_dir, f"{base_name}.pt")
        torch.save(hr_feats, out_path)
    print("Features extracted successfully")


def main_SAMOpenCLIP(data_dir: str, features_output_dir: str, sam_ckpt_path: str):
    model = OpenCLIPNetwork(OpenCLIPNetworkConfig)
    sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt_path).to("cuda")
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
    )

    img_folder = os.path.join(data_dir, "images")
    data_list = os.listdir(img_folder)
    data_list.sort()

    img_list = []
    WARNED = False
    for data_path in data_list:
        image_path = os.path.join(img_folder, data_path)
        image = cv2.imread(image_path)

        orig_w, orig_h = image.shape[1], image.shape[0]
        if orig_h > 1080:
            if not WARNED:
                print(
                    "[ INFO ] Encountered quite large input images (>1080P), rescaling to 1080P.\n "
                    "If this is not desired, please explicitly specify '--resolution/-r' as 1"
                )
                WARNED = True
            global_down = orig_h / 1080
        else:
            global_down = 1

        scale = float(global_down)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        image = cv2.resize(image, resolution)
        image = torch.from_numpy(image)
        img_list.append(image)
    images = [img_list[i].permute(2, 0, 1)[None, ...] for i in range(len(img_list))]
    imgs = torch.cat(images)

    os.makedirs(features_output_dir, exist_ok=True)
    create(
        imgs,
        data_list,
        features_output_dir,
        norm_clip_features=True,
        mask_generator=mask_generator,
        model=model,
    )


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    data_dir = args.source_path
    model_name = args.model
    features_output_dir = args.ouput_dir
    # Validate source directory
    if not os.path.isdir(data_dir):
        print(
            f"Error: The specified source directory '{data_dir}' does not exist or is not a directory."
        )
        exit(1)

    # Set device
    device = args.device
    print(f"Using device: {device}")
    if model_name == "SAMOpenCLIP":
        main_SAMOpenCLIP(data_dir, features_output_dir, args.sam_ckpt_path)
    else:
        main_featup(data_dir, features_output_dir, model_name, device)
