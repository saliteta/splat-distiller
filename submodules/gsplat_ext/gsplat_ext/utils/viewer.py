import argparse
import math
import os
import time
import typing
from dataclasses import dataclass
from abc import abstractmethod
import copy
import torch
import torch.nn.functional as F
import viser
from pathlib import Path
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from dataclasses import field
from nerfview import CameraState, RenderTabState, apply_float_colormap
from sklearn.decomposition import PCA
from .primitives import Primitive, GaussianPrimitive, DrSplatPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from .renderer import GaussianRenderer, GaussianRenderer2D, BetaSplatRenderer
from .text_encoder import TextEncoder
from typing import List, Union, Literal, Tuple
from nerfview import Viewer, RenderTabState


@dataclass
class ViewerState(RenderTabState):
    backgrounds: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    render_mode: Literal["RGB", "Feature", "AttentionMap"] = "RGB"
    positive_query: str = ""
    negative_query: List[str] = field(default_factory=lambda: ["background", "texture", "object"])
    threshold: Union[float, None] = 0.0
    query_features: torch.Tensor = None



class GeneralViewer(Viewer):
    def __init__(self, server: viser.ViserServer, splat_path: Path, splat_method: Literal["3DGS", "2DGS", "DBS"], 
                    feature_path: Path, text_encoder: TextEncoder, viewer_state: ViewerState, port=8080):
        self.feature_path = feature_path
        self.splat_path = splat_path
        self.splat_method = splat_method
        self.text_encoder = text_encoder
        self.port = port
        self.viewer_state = viewer_state
        self.render_tab_state = copy.deepcopy(viewer_state)
        super().__init__(server, self.render_function, None, 'rendering')
    
    def splat_method_selection(self):
        "set up splat entity"
        if self.splat_method == "3DGS":
            self.splat = GaussianPrimitive()
            self.splat.from_file(self.splat_path, self.feature_path)
            self.renderer = GaussianRenderer(self.splat)
            return "3D Gaussian Splatting"
        elif self.splat_method == "2DGS":
            self.splat = GaussianPrimitive2D()
            self.splat.from_file(self.splat_path, self.feature_path)
            self.renderer = GaussianRenderer2D(self.splat)
            return "2D Gaussian Splatting"
        elif self.splat_method == "DBS":
            self.splat = BetaSplatPrimitive()
            self.splat.from_file(self.splat_path, self.feature_path)
            self.renderer = BetaSplatRenderer(self.splat)
            return "Deformable Beta Splatting"
        else:
            raise ValueError(f"Unsupported splat method: {self.splat_method}")

    def reset(self):
        self.render_tab_state = copy.deepcopy(self.viewer_state)
        self.splat_method_selection()
        self.feature_pca()

    def feature_pca(self):
        features = self.splat.feature
        features_np = features.cpu().numpy()
        features_np = features_np.reshape(features_np.shape[0], -1)
        pca = PCA(n_components=3)
        features_pca = pca.fit_transform(features_np)
        features_pca = torch.from_numpy(features_pca).float().to('cuda')
        mins   = features_pca.min(dim=0).values    # shape (3,)
        maxs   = features_pca.max(dim=0).values    # shape (3,)
        
        # 2) compute range and add eps to avoid zero-div
        ranges = maxs - mins
        eps    = 1e-8
        
        # 3) normalize into [0,1]
        features_pca = (features_pca - mins) / (ranges + eps)
        self.features_pca = features_pca

    def camera_state_parser(self, camera_state: CameraState, render_tab_state: RenderTabState)-> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c2w = camera_state.c2w
        width = render_tab_state.render_width
        height = render_tab_state.render_height
        K = camera_state.get_K((width, height))
        c2w = torch.from_numpy(c2w).float().to('cuda').unsqueeze(0)
        K = torch.from_numpy(K).float().to('cuda').unsqueeze(0)
        return K, c2w, width, height

    @torch.no_grad()
    def render_function(self, camera_state: CameraState, render_tab_state: ViewerState):
        state = self.render_tab_state
        K, c2w, width, height = self.camera_state_parser(camera_state, render_tab_state)
        if state.render_mode == "RGB":
            image = self.renderer.render(K, c2w, width, height, state.render_mode)
        elif state.render_mode == "Feature":
            # Use PCA features for rendering, then restore original features
            original_feature = self.splat._feature
            self.splat._feature = self.features_pca  # Use PCA features for visualization
            image = self.renderer.render(K, c2w, width, height, state.render_mode)
            self.splat._feature = original_feature  # Restore original features
        elif state.render_mode == "AttentionMap":
            features = F.normalize(self.splat.feature, dim=1)
            sim = torch.sum( features * state.query_features, dim=-1, keepdim=True)  # [N,1]
            sim = sim.clamp(min=sim.mean())
            sim = (sim - sim.min()) / (sim.max() - sim.min())
            original_feature = self.splat._feature
            self.splat._feature = apply_float_colormap(sim, "turbo")
            image = self.renderer.render(K, c2w, width, height, "Feature")
            self.splat._feature = original_feature
        else:
            raise ValueError(f"Unsupported render mode: {state.render_mode}")
        return image.cpu().numpy()

    def query(self, query_text: List[str])-> torch.Tensor:
        # Ensure output is (b, c) where b = len(query_text), c = channel number
        query_features = [self.text_encoder.encode_text(text).cuda().float().flatten() for text in query_text]
        query_features = torch.stack(query_features, dim=0)  # (b, c)
        query_features = F.normalize(query_features, dim=1)
        return query_features


    def segmentation(self, positive_query: str, 
    negative_query: List[str] = ["background", "texture", "object"], 
    threshold: Union[float, None] = None) -> None:
        all_query_features = self.query([positive_query] + negative_query)
        key_features = self.splat.feature
        key_features = F.normalize(key_features, dim=1)
        # sim_matrix: (num_keys, num_queries)
        sim_matrix = torch.matmul(key_features, all_query_features.t())
        # mask: (num_keys, 1) -- True if positive query is the argmax
        mask = (sim_matrix.argmax(dim=1) == 0).float().unsqueeze(1)
        # If threshold is provided, also require that the positive similarity is above threshold
        if threshold is not None:
            pos_sim = sim_matrix[:, 0:1]
            mask = mask * (pos_sim > threshold).float()
        self.splat.mask(mask.bool().squeeze(1))
        self.feature_pca()


    def _init_rendering_tab(self):
        self._rendering_tab_handles = {}
        self._rendering_folder = self.server.gui.add_folder("Rendering")
    

    def _populate_rendering_tab(self):
        server = self.server
        with self._rendering_folder:
            with server.gui.add_folder(self.splat_method_selection()):
                self.feature_pca()

                self.reset_button = server.gui.add_button(
                    "Reset",
                    hint="Reset the viewer",
                )
                @self.reset_button.on_click
                def _(_) -> None:
                    self.reset()


                self.render_mode_dropdown = server.gui.add_dropdown(
                    "Render Mode",
                    (
                        "RGB",
                        "Feature",
                        "AttentionMap",
                    ),
                    initial_value=self.render_tab_state.render_mode,
                    hint="Render mode to use.",
                )
                ################ Positive Query ################
                positive_query_input = server.gui.add_text(
                    "Positive Query",
                    initial_value=self.render_tab_state.positive_query,
                    disabled=False,
                    hint="Use Relevance mode to query",
                )

                positive_query_submit_button = server.gui.add_button(
                    "Query",
                    disabled=False,
                    hint="Use Relevance mode to query",
                )

                @positive_query_submit_button.on_click
                def _(_) -> None:
                    self.render_tab_state.positive_query = positive_query_input.value
                    self.render_tab_state.text_change = True
                    if self.render_tab_state.render_mode == "AttentionMap":
                        self.render_tab_state.query_features = self.query([self.render_tab_state.positive_query])
                    self.rerender(_)
                ################ Positive Query ################

                ################ Negative Query ################
                negative_query_input = server.gui.add_text( 
                    "Negative Query",
                    initial_value=str(self.render_tab_state.negative_query),
                    disabled=False,
                    hint="Use Relevance mode to query",
                )

                negative_query_submit_button = server.gui.add_button(
                    "Segmentation",
                    disabled=False,
                    hint="Use Relevance mode to query",
                )

                @negative_query_submit_button.on_click
                def _(_) -> None:
                    self.render_tab_state.negative_query = negative_query_input.value.split(",")
                    self.render_tab_state.text_change = True
                    if self.render_tab_state.positive_query is None:
                        raise ValueError("Positive query is not set")
                    self.segmentation(self.render_tab_state.positive_query, self.render_tab_state.negative_query, self.render_tab_state.threshold)
                    self.rerender(_)
                ################ Negative Query ################

                ################ Threshold ################
                threshold_slider = server.gui.add_slider(
                    "Threshold",
                    initial_value=self.render_tab_state.threshold,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    hint="Threshold for segmentation.",
                )

                @threshold_slider.on_update
                def _(_) -> None:
                    self.render_tab_state.threshold = threshold_slider.value
                    self.segmentation(self.render_tab_state.positive_query, self.render_tab_state.negative_query, self.render_tab_state.threshold)
                    self.rerender(_)
                ################ Threshold ################


                @self.render_mode_dropdown.on_update
                def _(_) -> None:
                    self.render_tab_state.render_mode = self.render_mode_dropdown.value
                    if self.render_tab_state.render_mode == "AttentionMap":
                        self.render_tab_state.query_features = self.query([self.render_tab_state.positive_query])
                    self.rerender(_)



        self._rendering_tab_handles.update(
            {
                "render_mode_dropdown": self.render_mode_dropdown,
                "positive_query_input": positive_query_input,
                "negative_query_input": negative_query_input,
                "threshold_slider": threshold_slider,
                "backgrounds_slider": self.render_tab_state.backgrounds,
            }
        )

        super()._populate_rendering_tab()
                





