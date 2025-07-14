from gaussian_splatting.primitives import Primitive
from gaussian_splatting.primitives import GaussianPrimitive, GaussianPrimitive2D
from gsplat import rasterization, rasterization_2dgs
import torch
from typing import Literal


class Renderer:
    def __init__(self, primitives: Primitive):
        self.primitives = primitives

    def render(self, K, extrinsic, width, height, mode) -> torch.Tensor | None:
        pass


class GaussianRenderer(Renderer):
    def __init__(self, primitives: GaussianPrimitive):
        super().__init__(primitives)

    def render(
        self, K, extrinsic, width, height, mode: Literal["RGB", "Feature", "AttentionMap"], text_features: torch.Tensor | None = None
    ) -> torch.Tensor | None:
        if mode == "RGB":
            colors = self.primitives.color["colors"]
            renderedcolors, _, _ = rasterization(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                colors,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=2,
            )
            return renderedcolors.cpu().squeeze(0)

        elif mode == "Feature":
            assert self.primitives.feature is not None, "Feature is not available"
            features = self.primitives.feature
            renderedfeatures, _, _ = rasterization(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                features,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=None,
            )
            return renderedfeatures.cpu().squeeze(0)
        
        elif mode == "AttentionMap":
            assert self.primitives.feature is not None, "Feature is not available"
            assert text_features is not None, "Text features are not available"
            features = self.primitives.feature
            text_features = text_features.to(torch.float32)

            attention_scores = torch.einsum("nc,bc->nb", features, text_features) # N, len(text_features)
            renderedattention, _, _ = rasterization(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                attention_scores,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=None,
            )
            return renderedattention.cpu().squeeze(0)

        else:
            raise ValueError(f"Invalid mode: {mode}")


class GaussianRenderer2D(Renderer):
    def __init__(self, primitives: GaussianPrimitive2D):
        super().__init__(primitives)

    def render(self, K, extrinsic, width, height, mode: Literal["RGB", "Feature", "AttentionMap"], text_features: torch.Tensor | None = None) -> torch.Tensor | None:
        if mode == "RGB":
            colors = self.primitives.color["colors"]
            renderes = rasterization_2dgs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                colors,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=2,
            )
            rendered_colors = renderes[0]
            return rendered_colors.cpu().squeeze(0)
        
        elif mode == "Feature":
            assert self.primitives.feature is not None, "Feature is not available"
            features = self.primitives.feature
            renders = rasterization_2dgs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                features,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=None,
            )
            rendered_features = renders[0]
            return rendered_features.cpu().squeeze(0)
        
        elif mode == "AttentionMap":
            assert self.primitives.feature is not None, "Feature is not available"
            assert text_features is not None, "Text features are not available"
            features = self.primitives.feature
            text_features = text_features.to(torch.float32)

            attention_scores = torch.einsum("nc,bc->nb", features, text_features) # N, len(text_features)

            renders = rasterization_2dgs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                attention_scores,
                torch.linalg.inv(extrinsic),
                K,
                width,
                height,
                sh_degree=None,
            )
            rendered_attention = renders[0]
            return rendered_attention.cpu().squeeze(0)
        
        else:
            raise ValueError(f"Invalid mode: {mode}")