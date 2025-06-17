#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Literal, Tuple

import torch
import torch.nn.functional as F
from tqdm import trange

from nerfstudio.cameras.cameras import Cameras
from gsplat_ext.rasterization import inverse_rasterization_2dgs, inverse_rasterization_3dgs
from gs_loader.dataparser import (
    base_kernel_loader_config,
    general_gaussian_loader,
    general_cameras_loader,
)


def get_viewmat(c2w: torch.Tensor) -> torch.Tensor:
    R = c2w[:, :3, :3]
    T = c2w[:, :3, 3:4]
    R = R * torch.tensor([[[1, -1, -1]]], device=R.device, dtype=R.dtype)
    R_inv = R.transpose(1, 2)
    T_inv = -torch.bmm(R_inv, T)
    viewmat = torch.zeros(R.shape[0], 4, 4, device=R.device, dtype=R.dtype)
    viewmat[:, 3, 3] = 1.0
    viewmat[:, :3, :3] = R_inv
    viewmat[:, :3, 3:4] = T_inv
    return viewmat


def reverse_rasterization(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    image_features: torch.Tensor,
    camera: Cameras,
    mode: Literal["2DGS", "3DGS", "BSplats"]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    if mode == "2DGS":
        inverse_rasterization = inverse_rasterization_2dgs
    elif mode == "3DGS":
        inverse_rasterization = inverse_rasterization_3dgs
    elif mode == "BSplats":
        print("Contact Rong Liu, Butian Xiong")
        print("try to update to newest version")
        raise NotImplementedError
    else:
        print("we currently only support bsplat, 2DGS, and 3DGS")
        raise NotImplementedError


    viewmat = get_viewmat(camera.camera_to_worlds).cuda()
    K = camera.get_intrinsics_matrices().cuda()
    W, H = int(camera.width.item()), int(camera.height.item())

    feature_gaussian, feature_weight, gaussian_ids = inverse_rasterization_3dgs(
        means=means,
        quats=quats,
        scales=torch.exp(scales),
        opacities=torch.sigmoid(opacities).squeeze(-1),
        input_image=image_features,
        viewmats=viewmat,
        Ks=K,
        width=W,
        height=H,
        packed=True,
        render_mode='RGB',
    )

    return feature_gaussian, feature_weight, gaussian_ids


def frequency_filtering(scales, means, quats, opacities, threshold_size=1e7):
    if len(opacities) <= threshold_size:
        return means, quats, scales, opacities, None
    size = scales.mean(dim=1)
    _, top_indices = torch.topk(size, k=int(threshold_size))
    return (
        means[top_indices],
        quats[top_indices],
        scales[top_indices],
        opacities[top_indices],
        top_indices,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description="Compute & pack Gaussian features, with optional view-count stats."
    )
    p.add_argument('--data_location', type=str, required=True,
                   help="Dataset root (images + cameras)")
    p.add_argument('--data_mode', type=str, default='colmap',
                   choices=['colmap','scannet','nerfstudio','gsplat_colmap'],
                   help="Loader mode for cameras & features")
    p.add_argument('--pretrained_location', type=str, required=True,
                   help="Path to pretrained Gaussian model (.ckpt or .ply)")
    p.add_argument('--feature_location', type=str, required=True,
                   help="Folder of per-image feature .pt files")
    p.add_argument('--output_feature', type=str, required=True,
                   help="Where to save aggregated feature tensor .pt file")
    p.add_argument('--save_view_info', action='store_true',
                   help="Also compute per-Gaussian view counts and top indices")
    p.add_argument('--top_frac', type=float,
                   help="Fraction of Gaussians to keep by highest view counts")
    p.add_argument('--weights_file', type=str, default='feature_weights.pt',
                   help="Output .pt for view counts")
    p.add_argument('--id_file', type=str, default='feature_id.pt',
                   help="Output .pt for top indices by view count")
    p.add_argument('--mode', type=str, default='3DGS',
                   help="we can choose from 3DGS, 2DGS, and BSplats for now")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mode = args.mode

    # 1) load Gaussian geometry
    cfg = base_kernel_loader_config(
        kernel_location=args.pretrained_location,
        kernel_type=args.pretrained_location.split('.')[-1],
    )
    loader = general_gaussian_loader(cfg)
    geom = loader.load().geometry
    scales, means, quats, opacities = (
        geom['scales'], geom['means'], geom['quats'], geom['opacities']
    )
    means, quats, scales, opacities, indices = frequency_filtering(
        scales, means, quats, opacities
    )
    num_gaussians = len(opacities)

    # 2) data parser for real features
    data_parser = general_cameras_loader(
        data_path=Path(args.data_location),
        feature_path=args.feature_location,
        mode=args.data_mode,
    )

    # prepare accumulators
    features_gaussian = torch.zeros((num_gaussians,512), dtype=torch.float32).cuda()
    feature_weight   = torch.zeros((num_gaussians,1), dtype=torch.float32).cuda().squeeze()
    if args.save_view_info:
        view_counts = torch.zeros(num_gaussians, dtype=torch.long)

    # 3) loop over frames
    with torch.no_grad():
        for i in trange(len(data_parser), desc='Processing frames'):
            (camera, img_names, feature_location) = data_parser[i]
            camera = camera.to('cuda')
            feat_orig = torch.load(feature_location).cuda()
            feat = F.interpolate(
                feat_orig,
                size=(int(camera.height[0][0]), int(camera.width[0][0])),
                mode='bilinear', align_corners=False
            ).cuda()
            feat = F.normalize(feat.float(), dim=1).permute(0,2,3,1)
            del feat_orig; torch.cuda.empty_cache()

            fg, fw, gids = reverse_rasterization(
                means, quats, scales, opacities, feat, camera, mode
            )

            features_gaussian[gids] += fg
            feature_weight[gids]    += fw.squeeze(-1)
            if args.save_view_info:
                view_counts[torch.unique(gids).cpu()] += 1

    # -------------------------------------------------------------------------
    # FINAL SAVE LOGIC (modified for save_view_info behavior)
    # -------------------------------------------------------------------------
    safe_w = feature_weight.clone()
    safe_w[safe_w == 0] = 1
    features_gaussian /= safe_w.unsqueeze(1)

    # always save geometry subset indices if filtered
    if indices is not None:
        geom_id_file = Path(args.output_feature).stem + '_geom_id.pt'
        torch.save(indices, geom_id_file)
        print(f"Saved geometry subset indices → {geom_id_file}")

    if args.save_view_info and args.top_frac is not None:
        # a) save view counts
        torch.save(view_counts, args.weights_file)
        print(f"Saved view counts → {args.weights_file}")

        # b) pick top fraction and save their indices
        k = max(1, int(args.top_frac * num_gaussians))
        _, top_idx = torch.topk(view_counts, k=k, largest=True)
        torch.save(top_idx, args.id_file)
        print(f"Saved top {k} indices → {args.id_file}")

        # c) save only the features for those top-viewed Gaussians
        filtered_feats = features_gaussian[top_idx]
        torch.save(filtered_feats, args.output_feature)
        print(f"Saved top-viewed features → {args.output_feature}")
    else:
        # default: save all features
        torch.save(features_gaussian, args.output_feature)
        print(f"Saved all features → {args.output_feature}")