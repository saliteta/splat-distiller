'''
This is a general viewer for the splat primitives, it can be used to visualize the splat primitives in the scene.

It should be able to handle the following cases:
1. 2DGS
2. 3DGS
3. DBS


it can be easily extended to handle other splat primitives
it focus on the feature visualization, as well as the attention map visualization
it supports the segmentation using open vocabulary query
'''

from gsplat_ext import GeneralViewer, ViewerState, TextEncoder
from pathlib import Path
import torch
import viser
import time
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--feature_ckpt", type=str, required=True)
    parser.add_argument("--splat_method", type=str, required=True)
    parser.add_argument("--text_encoder", type=str, required=True)
    parser.add_argument("--port", type=int, required=False, default=8080)
    return parser.parse_args()

def main(args):
    splat_path = Path(args.ckpt)
    feature_path = Path(args.feature_ckpt)
    splat_method = args.splat_method

    device = torch.device("cuda")
    text_encoder = TextEncoder(args.text_encoder, device)

    server = viser.ViserServer(port=args.port, verbose=False)

    viewer = GeneralViewer(
        server=server,
        splat_path=splat_path,
        splat_method=splat_method,
        feature_path=feature_path,
        text_encoder=text_encoder,
        viewer_state=ViewerState(),
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)

if __name__ == "__main__":
    args = parse_args()
    main(args)