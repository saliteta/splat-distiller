#pragma once

#include <cstdint>
#include "Cameras.h"

namespace at {
class Tensor;
}

namespace gsplat_ext {

#define FILTER_INV_SQUARE_2DGS 2.0f

/////////////////////////////////////////////////
// rasterize_to_pixels_3dgs
/////////////////////////////////////////////////

/**
 * Launches the reverse 2D Gaussian rasterization CUDA kernel for a given channel dimension.
 *
 * @tparam CDIM Number of color/feature channels per pixel.
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param conics           Tensor of inverse of projected 2D Gayssuab
 * @param opacities        opacities of Gaussian: [C,N] or [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @param gauss_features   Output Gaussian features accumulator: [C, N, CDIM]
 * @param gauss_weights    Output Gaussian weight accumulator: [C, N]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 */
template <uint32_t CDIM>
void launch_reverse_rasterize_to_gaussians_3dgs_kernel(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &rendered_colors,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    at::Tensor       &gauss_features,
    at::Tensor       &gauss_weights,
    uint32_t          image_width,
    uint32_t          image_height,
    uint32_t          tile_size
);

/**
 * Launches the reverse 2D Gaussian rasterization CUDA kernel for a given channel dimension.
 *
 * @tparam CDIM Number of color/feature channels per pixel.
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param ray_transforms   Tensor of ray transform matrices: [C, N, 3, 3] or [nnz, 3, 3]
 * @param opacities        opacities of Gaussian: [C,N] or [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @param gauss_features   Output Gaussian features accumulator: [C, N, CDIM]
 * @param gauss_weights    Output Gaussian weight accumulator: [C, N]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 */
template <uint32_t CDIM>
void launch_reverse_rasterize_to_gaussians_2dgs_kernel(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &opacities,
    const at::Tensor &rendered_colors,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    at::Tensor       &gauss_features,
    at::Tensor       &gauss_weights,
    uint32_t          image_width,
    uint32_t          image_height,
    uint32_t          tile_size
);

/////////////////////////////////////////////////
// DBS
/////////////////////////////////////////////////


/**
 * Launches the reverse DBS Gaussian rasterization CUDA kernel for a given channel dimension.
 *
 * @tparam CDIM Number of color/feature channels per pixel.
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param conics           Tensor of inverse of projected 2D Gaussian conics: [C, N, 3] or [nnz, 3]
 * @param opacities        opacities of Gaussian: [C,N] or [nnz]
 * @param betas            Tensor of betas of Gaussian: [C, N] or [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @param gauss_features   Output Gaussian features accumulator: [C, N, CDIM]
 * @param gauss_weights    Output Gaussian weight accumulator: [C, N]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 */
template <uint32_t CDIM>
void launch_reverse_rasterize_to_gaussians_dbs_kernel(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &betas,
    const at::Tensor &rendered_colors,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids,
    at::Tensor       &gauss_features,
    at::Tensor       &gauss_weights,
    uint32_t          image_width,
    uint32_t          image_height,
    uint32_t          tile_size
);

} // namespace gsplat