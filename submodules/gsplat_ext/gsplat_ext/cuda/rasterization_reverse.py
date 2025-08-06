from typing import Optional, Tuple
from torch import Tensor
from ._backend import _C
import torch


def rasterize_reverse_fwd_2dgs(
    means2d: Tensor,
    ray_transforms: Tensor,
    opacities: Tensor,
    input_image: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    packed: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.
    This function is a cuda wrapper function that takes already processed information and feature information of images in,
    and output the feature of each 2D Gaussian influenced by that particular feature map.

    Args:
        means2d: Tensor, the center of each Gaussian on 2D map (nnz, 2)
        ray_transforms: 2D Gaussian's ray transformation matrix (nnz,3, 3)
        opacites Gaussian alpha (nnz)
        input_image: feature map (h,w,c)
        image_width: W
        image_height: H
        tile_size: the default size should be 16, every Gaussian is processed according to a tile
        tile_offset: previous processed result
        flatten_ids: previous processed result
        packed: will be test on packed is true
    Return:
        feature_gaussian: the feature accumulate for this particular image on the Gaussian it related to (nnz, n)
        feature_weight: The weight of each pixel combined (nnz) should be the same as the gradient of feature


    """
    C = isect_offsets.size(0)
    device = means2d.device
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert ray_transforms.shape == (nnz, 3, 3), ray_transforms.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert ray_transforms.shape == (C, N, 3, 3), ray_transforms.shape
        assert opacities.shape == (C, N), opacities.shape

    # Pad the channels to the nearest supported number if necessary
    channels = input_image.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")

    tile_height, tile_width = isect_offsets.shape[1:3]

    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    feature_gaussian, feature_weight = _C.rasterize_reverse_fwd_2dgs(
        means2d.contiguous(),
        ray_transforms.contiguous(),
        opacities.contiguous(),
        input_image.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )

    return feature_gaussian, feature_weight


def rasterize_reverse_fwd_3dgs(
    means2d: Tensor,
    conics: Tensor,
    opacities: Tensor,
    input_image: Tensor,
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,
    flatten_ids: Tensor,
    packed: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Rasterize Gaussians to pixels.
    This function is a cuda wrapper function that takes already processed information and feature information of images in,
    and output the feature of each 2D Gaussian influenced by that particular feature map.

    Args:
        means2d: Tensor, the center of each Gaussian on 2D map (nnz, 2)
        ray_transforms: 2D Gaussian's ray transformation matrix (nnz,3, 3)
        opacites Gaussian alpha (nnz)
        input_image: feature map (h,w,c)
        image_width: W
        image_height: H
        tile_size: the default size should be 16, every Gaussian is processed according to a tile
        tile_offset: previous processed result
        flatten_ids: previous processed result
        packed: will be test on packed is true
    Return:
        feature_gaussian: the feature accumulate for this particular image on the Gaussian it related to (nnz, n)
        feature_weight: The weight of each pixel combined (nnz) should be the same as the gradient of feature


    """
    C = isect_offsets.size(0)
    if packed:
        nnz = means2d.size(0)
        assert means2d.shape == (nnz, 2), means2d.shape
        assert conics.shape == (nnz, 3), conics.shape
        assert opacities.shape == (nnz,), opacities.shape
    else:
        N = means2d.size(1)
        assert means2d.shape == (C, N, 2), means2d.shape
        assert conics.shape == (C, N, 3), conics.shape
        assert opacities.shape == (C, N), opacities.shape

    # Pad the channels to the nearest supported number if necessary
    channels = input_image.shape[-1]
    if channels > 512 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")

    tile_height, tile_width = isect_offsets.shape[1:3]

    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    feature_gaussian, feature_weight = _C.rasterize_reverse_fwd_3dgs(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        input_image.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )

    return feature_gaussian, feature_weight


def rasterize_reverse_fwd_dbs(
    means2d: Tensor,  # [C, N, 2] or [nnz, 2]
    conics: Tensor,  # [C, N, 3] or [nnz, 3]
    opacities: Tensor,  # [C, N] or [nnz]
    betas: Tensor,  # [C, N] or [nnz]
    rendered_colors: Tensor,  # [C, H, W, CDIM]
    image_width: int,
    image_height: int,
    tile_size: int,
    isect_offsets: Tensor,  # [C, tile_height, tile_width]
    flatten_ids: Tensor,  # [n_isects]
) -> Tuple[Tensor, Tensor]:
    """Rasterizes Gaussians to pixels. Our default is packed.

    Args:
        means2d: Projected Gaussian means. [C, N, 2] if packed is False, [nnz, 2] if packed is True.
        conics: Inverse of the projected covariances with only upper triangle values. [C, N, 3] if packed is False, [nnz, 3] if packed is True.
        colors: Gaussian colors or ND features. [C, N, channels] if packed is False, [nnz, channels] if packed is True.
        opacities: Gaussian opacities that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        betas: Gaussian sharpness that support per-view values. [C, N] if packed is False, [nnz] if packed is True.
        image_width: Image width.
        image_height: Image height.
        tile_size: Tile size.
        isect_offsets: Intersection offsets outputs from `isect_offset_encode()`. [C, tile_height, tile_width]
        flatten_ids: The global flatten indices in [C * N] or [nnz] from  `isect_tiles()`. [n_isects]

    Returns:
        A tuple:

        - **Rendered colors**. [C, image_height, image_width, channels]
        - **Rendered alphas**. [C, image_height, image_width, 1]
    """

    C = isect_offsets.size(0)
    device = means2d.device
    nnz = means2d.size(0)
    assert means2d.shape == (nnz, 2), means2d.shape
    assert conics.shape == (nnz, 3), conics.shape
    assert opacities.shape == (nnz,), opacities.shape
    assert betas.shape == (nnz,), betas.shape

    # Pad the channels to the nearest supported number if necessary
    channels = rendered_colors.shape[-1]
    if channels > 513 or channels == 0:
        # TODO: maybe worth to support zero channels?
        raise ValueError(f"Unsupported number of color channels: {channels}")
    if channels not in (
        1,
        2,
        3,
        4,
        5,
        8,
        9,
        16,
        17,
        32,
        33,
        64,
        65,
        128,
        129,
        256,
        257,
        512,
        513,
    ):
        padded_channels = (1 << (channels - 1).bit_length()) - channels
        rendered_colors = torch.cat(
            [
                rendered_colors,
                torch.zeros(
                    *rendered_colors.shape[:-1], padded_channels, device=device
                ),
            ],
            dim=-1,
        )
        padded_channels = 0

    tile_height, tile_width = isect_offsets.shape[1:3]
    assert (
        tile_height * tile_size >= image_height
    ), f"Assert Failed: {tile_height} * {tile_size} >= {image_height}"
    assert (
        tile_width * tile_size >= image_width
    ), f"Assert Failed: {tile_width} * {tile_size} >= {image_width}"

    feature_gaussian, feature_weight = _C.rasterize_reverse_fwd_dbs(
        means2d.contiguous(),
        conics.contiguous(),
        opacities.contiguous(),
        betas.contiguous(),
        rendered_colors.contiguous(),
        image_width,
        image_height,
        tile_size,
        isect_offsets.contiguous(),
        flatten_ids.contiguous(),
    )

    return feature_gaussian, feature_weight
