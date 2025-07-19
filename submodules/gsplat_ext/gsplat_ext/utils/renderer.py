from .primitives import Primitive
from .primitives import GaussianPrimitive, GaussianPrimitive2D, BetaSplatPrimitive
from gsplat import rasterization, rasterization_2dgs
from ..rasterization import (
    inverse_rasterization_2dgs,
    inverse_rasterization_3dgs,
    inverse_rasterization_dbs,
)
from bsplat import rasterization as rasterization_dbs
import torch
from typing import Literal, Tuple


class Renderer:
    def __init__(self, primitives: Primitive):
        self.primitives = primitives

    def render(self, K, extrinsic, width, height, mode) -> torch.Tensor | None:
        pass

    def inverse_render(
        self, K, extrinsic, width, height, features: torch.Tensor
    ) -> torch.Tensor | None:
        pass


class GaussianRenderer(Renderer):
    def __init__(self, primitives: GaussianPrimitive):
        super().__init__(primitives)

    def render(
        self,
        K,
        extrinsic,
        width,
        height,
        mode: Literal["RGB", "Feature", "AttentionMap"],
        text_features: torch.Tensor | None = None,
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

            attention_scores = torch.einsum(
                "nc,bc->nb", features, text_features
            )  # N, len(text_features)
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

    def inverse_render(
        self, K, extrinsic, width, height, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inverse render the features to the primitives.
        """
        gaussian_features, gaussian_weights, primitive_ids = inverse_rasterization_3dgs(
            means=self.primitives.geometry["means"],
            quats=self.primitives.geometry["quats"],
            scales=self.primitives.geometry["scales"],
            opacities=self.primitives.geometry["opacities"],
            input_image=features,
            viewmats=torch.linalg.inv(extrinsic),
            Ks=K,
            width=width,
            height=height,
        )
        return gaussian_features, gaussian_weights, primitive_ids


class GaussianRenderer2D(Renderer):
    def __init__(self, primitives: GaussianPrimitive2D):
        super().__init__(primitives)

    def render(
        self,
        K,
        extrinsic,
        width,
        height,
        mode: Literal["RGB", "Feature", "AttentionMap"],
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
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

            attention_scores = torch.einsum(
                "nc,bc->nb", features, text_features
            )  # N, len(text_features)

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

    def inverse_render(
        self, K, extrinsic, width, height, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inverse render the features to the primitives.
        """
        gaussian_features, gaussian_weights, primitive_ids = inverse_rasterization_2dgs(
            means=self.primitives.geometry["means"],
            quats=self.primitives.geometry["quats"],
            scales=self.primitives.geometry["scales"],
            opacities=self.primitives.geometry["opacities"],
            input_image=features,
            viewmats=torch.linalg.inv(extrinsic),
            Ks=K,
            width=width,
            height=height,
        )
        return gaussian_features, gaussian_weights, primitive_ids


class BetaSplatRenderer(Renderer):
    def __init__(self, primitives: BetaSplatPrimitive):
        super().__init__(primitives)

    def inverse_render(
        self, K, extrinsic, width, height, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inverse render the features to the primitives.
        """
        gaussian_features, gaussian_weights, primitive_ids = inverse_rasterization_dbs(
            means=self.primitives.geometry["means"],
            quats=self.primitives.geometry["quats"],
            scales=self.primitives.geometry["scales"],
            opacities=self.primitives.geometry["opacities"],
            betas=self.primitives.color["beta"],
            rendered_colors=features,
            viewmats=torch.linalg.inv(extrinsic),
            Ks=K,
            width=width,
            height=height,
        )
        return gaussian_features, gaussian_weights, primitive_ids

    def render(
        self,
        K,
        extrinsic,
        width,
        height,
        mode: Literal["RGB", "Feature", "AttentionMap"],
        text_features: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        assert isinstance(
            self.primitives.geometry["sh_degree"], int
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.geometry["sb_number"], int
        ), "SB number is not available"
        assert isinstance(
            self.primitives.geometry["means"], torch.Tensor
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.geometry["quats"], torch.Tensor
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.geometry["scales"], torch.Tensor
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.geometry["opacities"], torch.Tensor
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.color["beta"], torch.Tensor
        ), "SH degree is not available"
        assert isinstance(
            self.primitives.color["sh0"], torch.Tensor
        ), "SH degree is not available"

        if mode == "RGB":
            rendered_colors, _, _ = rasterization_dbs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                betas=self.primitives.color["beta"],
                colors=self.primitives.color["sh0"],
                viewmats=torch.linalg.inv(extrinsic),
                Ks=K,
                width=width,
                height=height,
                sh_degree=self.primitives.geometry["sh_degree"],
                sb_number=self.primitives.geometry["sb_number"],
                sb_params=self.primitives.color["sb_params"],
                backgrounds=torch.zeros(1, 3).cuda(),
                packed=True,
            )
            return rendered_colors.cpu().squeeze(0)

        elif mode == "Feature":
            assert self.primitives.feature is not None, "Feature is not available"
            rendered_features, _, _ = rasterization_dbs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                betas=self.primitives.color["beta"],
                colors=self.primitives.feature,
                viewmats=torch.linalg.inv(extrinsic),
                Ks=K,
                width=width,
                height=height,
                sh_degree=None,
                sb_number=None,
                packed=True,
            )
            return rendered_features.cpu().squeeze(0)

        elif mode == "AttentionMap":
            assert self.primitives.feature is not None, "Feature is not available"
            assert text_features is not None, "Text features are not available"
            features = self.primitives.feature
            text_features = text_features.to(torch.float32)

            attention_scores = torch.einsum(
                "nc,bc->nb", features, text_features
            )  # N, len(text_features)

            rendered_attention, _, _ = rasterization_dbs(
                self.primitives.geometry["means"],
                self.primitives.geometry["quats"],
                self.primitives.geometry["scales"],
                self.primitives.geometry["opacities"],
                betas=self.primitives.color["beta"],
                colors=attention_scores,
                viewmats=torch.linalg.inv(extrinsic),
                Ks=K,
                width=width,
                height=height,
                sh_degree=None,
                sb_number=None,
            )
            return rendered_attention.cpu().squeeze(0)


__all__ = ["Renderer", "GaussianRenderer", "GaussianRenderer2D", "BetaSplatRenderer"]
