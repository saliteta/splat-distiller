import argparse
import math
import os
import time

import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization

from nerfview import CameraState, RenderTabState, apply_float_colormap
from gsplat_viewer import GsplatViewer, GsplatRenderTabState
from sklearn.decomposition import PCA
from featup.featurizers.maskclip.clip import tokenize


def main(local_rank: int, world_rank, world_size: int, args):
    torch.manual_seed(42)
    device = torch.device("cuda", local_rank)
    clip_model = (
        torch.hub.load("mhamilton723/FeatUp", "maskclip", use_norm=False)
        .to(device)
        .eval()
        .model.model
    )

    ckpt = torch.load(args.ckpt, map_location=device)["splats"]
    means = ckpt["means"]
    quats = F.normalize(ckpt["quats"], p=2, dim=-1)
    scales = torch.exp(ckpt["scales"])
    opacities = torch.sigmoid(ckpt["opacities"])
    sh0 = ckpt["sh0"]
    shN = ckpt["shN"]
    colors = torch.cat([sh0, shN], dim=-2)
    sh_degree = int(math.sqrt(colors.shape[-2]) - 1)
    print("Number of Gaussians:", len(means))

    features = None
    features_pca = None

    if args.feature_ckpt is None:
        args.feature_ckpt = os.path.splitext(args.ckpt)[0] + "_features.pt"
    if os.path.exists(args.feature_ckpt):
        features = torch.load(args.feature_ckpt, map_location=device)
        print("Using features from", args.feature_ckpt)
        features_np = features.cpu().numpy()
        features_np = features_np.reshape(features_np.shape[0], -1)
        # Perform PCA to reduce the feature dimensions to 3
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_np)
        features_pca = torch.from_numpy(features_pca).float().to(device)
        mins   = features_pca.min(dim=0).values    # shape (3,)
        maxs   = features_pca.max(dim=0).values    # shape (3,)
        
        # 2) compute range and add eps to avoid zero-div
        ranges = maxs - mins
        eps    = 1e-8
        
        # 3) normalize into [0,1]
        features_pca = (features_pca - mins) / (ranges + eps)
        
    @torch.no_grad()
    def compute_relevance(features, render_tab_state):
        text_features = clip_model.encode_text(
            tokenize(render_tab_state.query_text).cuda()
        ).float()
        text_features = F.normalize(text_features, dim=0)
        features = F.normalize(features, dim=0)

        # Cosine similarity
        sim = torch.sum(features * text_features, dim=-1, keepdim=True)  # [N,1]
        sim = sim.clamp(min=sim.mean())
        sim = (sim - sim.min()) / (sim.max() - sim.min())
        return apply_float_colormap(sim, render_tab_state.colormap)

    # register and open viewer
    @torch.no_grad()
    def viewer_render_fn(camera_state: CameraState, render_tab_state: RenderTabState):
        assert isinstance(render_tab_state, GsplatRenderTabState)
        if render_tab_state.preview_render:
            width = render_tab_state.render_width
            height = render_tab_state.render_height
        else:
            width = render_tab_state.viewer_width
            height = render_tab_state.viewer_height

        if render_tab_state.text_change and features is not None:
            render_tab_state.relevance = compute_relevance(features, render_tab_state)
            render_tab_state.text_change = False

        c2w = camera_state.c2w
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to(device)
        K = torch.from_numpy(K).float().to(device)
        viewmat = c2w.inverse()

        RENDER_MODE_MAP = {
            "rgb": "RGB",
            "depth(accumulated)": "D",
            "depth(expected)": "ED",
            "alpha": "RGB",
            "diffuse": "Diffuse",
            "specular": "Specular",
            "feature": "RGB",
            "relevance": "RGB",
        }

        render_colors, render_alphas, info = rasterization(
            means,  # [N, 3]
            quats,  # [N, 4]
            scales,  # [N, 3]
            opacities,  # [N]
            (
                features_pca
                if render_tab_state.render_mode == "feature"
                else render_tab_state.relevance
                if render_tab_state.render_mode == "relevance"
                else colors
            ),  # [N, S, 3]
            viewmat[None],  # [1, 4, 4]
            K[None],  # [1, 3, 3]
            width,
            height,
            sh_degree=(
                None
                if render_tab_state.render_mode in ["feature", "relevance"]
                else (
                    min(render_tab_state.max_sh_degree, sh_degree)
                    if sh_degree is not None
                    else None
                )
            ),
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
            eps2d=render_tab_state.eps2d,
            backgrounds=torch.tensor([render_tab_state.backgrounds], device=device)
            / 255.0,
            render_mode=RENDER_MODE_MAP[render_tab_state.render_mode],
            rasterize_mode=render_tab_state.rasterize_mode,
            camera_model=render_tab_state.camera_model,
            packed=False,
            with_ut=args.with_ut,
            with_eval3d=args.with_eval3d,
        )
        render_tab_state.total_gs_count = len(means)
        render_tab_state.rendered_gs_count = (info["radii"] > 0).all(-1).sum().item()

        if render_tab_state.render_mode in ["depth(accumulated)", "depth(expected)"]:
            # normalize depth to [0, 1]
            depth = render_colors[0, ..., 0:1]
            if render_tab_state.normalize_nearfar:
                near_plane = render_tab_state.near_plane
                far_plane = render_tab_state.far_plane
            else:
                near_plane = depth.min()
                far_plane = depth.max()
            depth_norm = (depth - near_plane) / (far_plane - near_plane + 1e-10)
            depth_norm = torch.clip(depth_norm, 0, 1)
            if render_tab_state.inverse:
                depth_norm = 1 - depth_norm
            renders = (
                apply_float_colormap(depth_norm, render_tab_state.colormap)
                .cpu()
                .numpy()
            )
        elif render_tab_state.render_mode == "alpha":
            alpha = render_alphas[0, ..., 0:1]
            renders = (
                apply_float_colormap(alpha, render_tab_state.colormap).cpu().numpy()
            )
        else:
            # colors represented with sh are not guranteed to be in [0, 1]
            render_colors = render_colors[0, ..., 0:3].clamp(0, 1)
            renders = render_colors.cpu().numpy()
        return renders

    server = viser.ViserServer(port=args.port, verbose=False)

    viewer = GsplatViewer(
        server=server,
        render_fn=viewer_render_fn,
        output_dir=Path(args.output_dir),
        mode="rendering",
    )
    if features is None:
        viewer.render_mode_dropdown.options = (
            "rgb",
            "depth(accumulated)",
            "depth(expected)",
            "alpha",
            "diffuse",
            "specular",
        )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", type=str, default="results/", help="where to dump outputs"
    )
    parser.add_argument(
        "--scene_grid", type=int, default=1, help="repeat the scene into a grid of NxN"
    )
    parser.add_argument("--ckpt", type=str, default=None, help="path to the .pt file")
    parser.add_argument(
        "--feature_ckpt", type=str, default=None, help="path to the features.pt file"
    )
    parser.add_argument(
        "--port", type=int, default=8080, help="port for the viewer server"
    )
    parser.add_argument(
        "--with_ut", action="store_true", help="use uncentered transform"
    )
    parser.add_argument("--with_eval3d", action="store_true", help="use eval 3D")
    args = parser.parse_args()
    assert args.scene_grid % 2 == 1, "scene_grid must be odd"

    cli(main, args, verbose=True)
