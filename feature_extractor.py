import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

# Supported models
SUPPORTED_MODELS = ["dino16", "dinov2", "clip", "maskclip", "vit", "resnet50"]


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

    parser.add_argument("--device", type=str, default="cuda", help="pytorch device")

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


def main():
    # Parse command-line arguments
    args = parse_arguments()
    data_dir = args.source_path
    features_output_dir = os.path.join(data_dir, "features")
    model_name = args.model.lower()

    # Validate source directory
    if not os.path.isdir(data_dir):
        print(
            f"Error: The specified source directory '{data_dir}' does not exist or is not a directory."
        )
        exit(1)

    # Set device
    device = args.device
    print(f"Using device: {device}")

    # Define transformations
    transform = T.Compose(
        [
            T.Resize(224),
            T.CenterCrop(224),
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


if __name__ == "__main__":
    main()
