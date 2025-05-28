#pragma once

// old gsplat
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

// new gsplat
#include <ATen/core/Tensor.h>
#include "Cameras.h"
#include "Common.h"

#define GSPLAT_N_THREADS 256

#define GSPLAT_CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define GSPLAT_CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define GSPLAT_CHECK_INPUT(x)                                                  \
    GSPLAT_CHECK_CUDA(x);                                                      \
    GSPLAT_CHECK_CONTIGUOUS(x)
#define GSPLAT_DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define GSPLAT_PRAGMA_UNROLL _Pragma("unroll")

// https://github.com/pytorch/pytorch/blob/233305a852e1cd7f319b15b5137074c9eac455f6/aten/src/ATen/cuda/cub.cuh#L38-L46
#define GSPLAT_CUB_WRAPPER(func, ...)                                          \
    do {                                                                       \
        size_t temp_storage_bytes = 0;                                         \
        func(nullptr, temp_storage_bytes, __VA_ARGS__);                        \
        auto &caching_allocator = *::c10::cuda::CUDACachingAllocator::get();   \
        auto temp_storage = caching_allocator.allocate(temp_storage_bytes);    \
        func(temp_storage.get(), temp_storage_bytes, __VA_ARGS__);             \
    } while (false)

namespace gsplat_ext {


/**
 * High-level C++ binding for reverse 3D Gaussian rasterization.
 * Calls the above kernel launcher based on the CDIM.
 *
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param conics   Tensor of opacities of Gaussian: [C, N] or [nnz]
 * @param opacities        Opacities:  [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @return A pair of tensors: (gauss_features: [C, N, CDIM], gauss_weights: [C, N])
 */
std::tuple<torch::Tensor, torch::Tensor>
reverse_rasterize_to_gaussians_3dgs(
    const at::Tensor &means2d,
    const at::Tensor &conics,
    const at::Tensor &opacities,
    const at::Tensor &rendered_colors,
    uint32_t          image_width,
    uint32_t          image_height,
    uint32_t          tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
);



// 2D GS newest GSPLAT
/**
 * High-level C++ binding for reverse 2D Gaussian rasterization.
 * Calls the above kernel launcher based on the CDIM.
 *
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param ray_transforms   Tensor of ray transform matrices: [C, N, 3, 3] or [nnz, 3, 3]
 * @param opacities        Opacities:  [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @return A pair of tensors: (gauss_features: [C, N, CDIM], gauss_weights: [C, N])
 */
std::tuple<torch::Tensor, torch::Tensor>
reverse_rasterize_to_gaussians_2dgs(
    const at::Tensor &means2d,
    const at::Tensor &ray_transforms,
    const at::Tensor &opacities,
    const at::Tensor &rendered_colors,
    uint32_t          image_width,
    uint32_t          image_height,
    uint32_t          tile_size,
    const at::Tensor &tile_offsets,
    const at::Tensor &flatten_ids
);



// 2D GS newest GSPLAT
/**
 * High-level C++ binding for reverse 2D Gaussian rasterization.
 * Calls the above kernel launcher based on the CDIM.
 *
 * @param means2d          Tensor of Gaussian centers: [C, N, 2] or packed [nnz, 2]
 * @param conics   Tensor of ray transform matrices: [C, N, 3, 3] or [nnz, 3, 3]
 * @param opacities        Opacities:  [nnz]
 * @param betas        Opacities:  [nnz]
 * @param rendered_colors  Rendered image tensor: [C, H, W, CDIM]
 * @param image_width      Width of the rendered image
 * @param image_height     Height of the rendered image
 * @param tile_size        Tile size for block dispatch
 * @param tile_offsets     Intersection tile offsets: [C, tile_h, tile_w]
 * @param flatten_ids      Flattened Gaussian indices: [n_isects]
 * @return A pair of tensors: (gauss_features: [C, N, CDIM], gauss_weights: [C, N])
 */
std::tuple<torch::Tensor, torch::Tensor>
reverse_rasterize_to_bsplats(
    // Gaussian parameters
    const at::Tensor& means2d,   // [C, N, 2] or [nnz, 2]
    const at::Tensor& conics,    // [C, N, 3] or [nnz, 3]
    const at::Tensor& opacities, // [C, N]  or [nnz]
    const at::Tensor& betas, // [C, N]  or [nnz]
    const at::Tensor& rendered_colors, // [C, H, W, CDIM]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor& tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor& flatten_ids   // [n_isects]
);

} // namespace gsplat_ext

