#include <ATen/TensorUtils.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAGuard.h> // for DEVICE_GUARD
#include <tuple>

#include <ATen/Functions.h>
#include <ATen/NativeFunctions.h>

#include "Common.h"
#include "Rasterization.h"
#include "Cameras.h"
#include "Ops.h"

namespace gsplat_ext {
////////////////////////////////////////////////////
// 3DGS
////////////////////////////////////////////////////

std::tuple<at::Tensor, at::Tensor> reverse_rasterize_to_gaussians_3dgs(
    // Gaussian parameters
    const at::Tensor& means2d,   // [C, N, 2] or [nnz, 2]
    const at::Tensor& conics,    // [C, N, 3] or [nnz, 3]
    const at::Tensor& opacities, // [C, N]  or [nnz]
    const at::Tensor& rendered_colors, // [C, H, W, CDIM]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor& tile_offsets, // [C, tile_height, tile_width]
    const at::Tensor& flatten_ids   // [n_isects]
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt = means2d.options();
    uint32_t CDIM     = rendered_colors.size(-1);

    // allocate output tensors we only deal with packed equals to true
    at::Tensor gauss_features = at::zeros({means2d.size(0), CDIM}, opt);
    at::Tensor gauss_weights  = at::zeros({means2d.size(0)},     opt);

#define __LAUNCH_REVERSE_KERNEL__(K)                                           \
    case K:                                                                     \
        launch_reverse_rasterize_to_gaussians_3dgs_kernel<K>(                   \
            means2d,                                                           \
            conics,                                                      \
            opacities,                                                      \
            rendered_colors,                                                     \
            tile_offsets,                                                        \
            flatten_ids,                                                         \
            gauss_features,                                                      \
            gauss_weights,                                                       \
            image_width,                                                         \
            image_height,                                                        \
            tile_size                                                           \
        );                                                                      \
        break;

    switch (CDIM) {
        __LAUNCH_REVERSE_KERNEL__(1)
        __LAUNCH_REVERSE_KERNEL__(2)
        __LAUNCH_REVERSE_KERNEL__(3)
        __LAUNCH_REVERSE_KERNEL__(4)
        __LAUNCH_REVERSE_KERNEL__(5)
        __LAUNCH_REVERSE_KERNEL__(8)
        __LAUNCH_REVERSE_KERNEL__(9)
        __LAUNCH_REVERSE_KERNEL__(16)
        __LAUNCH_REVERSE_KERNEL__(17)
        __LAUNCH_REVERSE_KERNEL__(32)
        __LAUNCH_REVERSE_KERNEL__(33)
        __LAUNCH_REVERSE_KERNEL__(64)
        __LAUNCH_REVERSE_KERNEL__(65)
        __LAUNCH_REVERSE_KERNEL__(128)
        __LAUNCH_REVERSE_KERNEL__(129)
        __LAUNCH_REVERSE_KERNEL__(256)
        __LAUNCH_REVERSE_KERNEL__(257)
        __LAUNCH_REVERSE_KERNEL__(512)
        __LAUNCH_REVERSE_KERNEL__(513)
    default:
        AT_ERROR("Unsupported channel dimension for reverse rasterize: ", CDIM);
    }
#undef __LAUNCH_REVERSE_KERNEL__

    return std::make_tuple(gauss_features, gauss_weights);
}



////////////////////////////////////////////////////
// 2DGS
////////////////////////////////////////////////////

std::tuple<at::Tensor, at::Tensor> reverse_rasterize_to_gaussians_2dgs(
    const at::Tensor& means2d,         // [C, N, 2] or [nnz, 2]
    const at::Tensor& ray_transforms,  // [C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor& opacities,  // [C, N, 3, 3] or [nnz, 3, 3]
    const at::Tensor& rendered_colors, // [C, H, W, CDIM]
    const uint32_t     image_width,
    const uint32_t     image_height,
    const uint32_t     tile_size,
    const at::Tensor& tile_offsets,    // [C, tile_h, tile_w]
    const at::Tensor& flatten_ids      // [n_isects]
) {
    // Ensure all inputs are on the same CUDA device
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(ray_transforms);
    CHECK_INPUT(opacities);
    CHECK_INPUT(rendered_colors);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt = means2d.options();
    uint32_t CDIM     = rendered_colors.size(-1);

    // allocate output tensors we only deal with packed equals to true
    at::Tensor gauss_features = at::zeros({means2d.size(0), CDIM}, opt);
    at::Tensor gauss_weights  = at::zeros({means2d.size(0)},     opt);

    // dispatch by channel dimension
#define __LAUNCH_REVERSE_KERNEL__(K)                                           \
    case K:                                                                     \
        launch_reverse_rasterize_to_gaussians_2dgs_kernel<K>(                   \
            means2d,                                                           \
            ray_transforms,                                                      \
            opacities,                                                      \
            rendered_colors,                                                     \
            tile_offsets,                                                        \
            flatten_ids,                                                         \
            gauss_features,                                                      \
            gauss_weights,                                                       \
            image_width,                                                         \
            image_height,                                                        \
            tile_size                                                           \
        );                                                                      \
        break;

    switch (CDIM) {
        __LAUNCH_REVERSE_KERNEL__(1)
        __LAUNCH_REVERSE_KERNEL__(2)
        __LAUNCH_REVERSE_KERNEL__(3)
        __LAUNCH_REVERSE_KERNEL__(4)
        __LAUNCH_REVERSE_KERNEL__(5)
        __LAUNCH_REVERSE_KERNEL__(8)
        __LAUNCH_REVERSE_KERNEL__(9)
        __LAUNCH_REVERSE_KERNEL__(16)
        __LAUNCH_REVERSE_KERNEL__(17)
        __LAUNCH_REVERSE_KERNEL__(32)
        __LAUNCH_REVERSE_KERNEL__(33)
        __LAUNCH_REVERSE_KERNEL__(64)
        __LAUNCH_REVERSE_KERNEL__(65)
        __LAUNCH_REVERSE_KERNEL__(128)
        __LAUNCH_REVERSE_KERNEL__(129)
        __LAUNCH_REVERSE_KERNEL__(256)
        __LAUNCH_REVERSE_KERNEL__(257)
        __LAUNCH_REVERSE_KERNEL__(512)
        __LAUNCH_REVERSE_KERNEL__(513)
    default:
        AT_ERROR("Unsupported channel dimension for reverse rasterize: ", CDIM);
    }
#undef __LAUNCH_REVERSE_KERNEL__

    return std::make_tuple(gauss_features, gauss_weights);
}


////////////////////////////////////////////////////
// DBS
////////////////////////////////////////////////////


std::tuple<at::Tensor, at::Tensor> reverse_rasterize_to_gaussians_dbs(
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
) {
    DEVICE_GUARD(means2d);
    CHECK_INPUT(means2d);
    CHECK_INPUT(conics);
    CHECK_INPUT(opacities);
    CHECK_INPUT(betas);
    CHECK_INPUT(tile_offsets);
    CHECK_INPUT(flatten_ids);

    auto opt = means2d.options();
    uint32_t CDIM     = rendered_colors.size(-1);

    // allocate output tensors we only deal with packed equals to true
    at::Tensor gauss_features = at::zeros({means2d.size(0), CDIM}, opt);
    at::Tensor gauss_weights  = at::zeros({means2d.size(0)},     opt);

#define __LAUNCH_REVERSE_KERNEL__(K)                                           \
    case K:                                                                     \
        launch_reverse_rasterize_to_gaussians_dbs_kernel<K>(                   \
            means2d,                                                           \
            conics,                                                      \
            opacities,                                                      \
            betas,                                                      \
            rendered_colors,                                                     \
            tile_offsets,                                                        \
            flatten_ids,                                                         \
            gauss_features,                                                      \
            gauss_weights,                                                       \
            image_width,                                                         \
            image_height,                                                        \
            tile_size                                                           \
        );                                                                      \
        break;

    switch (CDIM) {
        __LAUNCH_REVERSE_KERNEL__(1)
        __LAUNCH_REVERSE_KERNEL__(2)
        __LAUNCH_REVERSE_KERNEL__(3)
        __LAUNCH_REVERSE_KERNEL__(4)
        __LAUNCH_REVERSE_KERNEL__(5)
        __LAUNCH_REVERSE_KERNEL__(8)
        __LAUNCH_REVERSE_KERNEL__(9)
        __LAUNCH_REVERSE_KERNEL__(16)
        __LAUNCH_REVERSE_KERNEL__(17)
        __LAUNCH_REVERSE_KERNEL__(32)
        __LAUNCH_REVERSE_KERNEL__(33)
        __LAUNCH_REVERSE_KERNEL__(64)
        __LAUNCH_REVERSE_KERNEL__(65)
        __LAUNCH_REVERSE_KERNEL__(128)
        __LAUNCH_REVERSE_KERNEL__(129)
        __LAUNCH_REVERSE_KERNEL__(256)
        __LAUNCH_REVERSE_KERNEL__(257)
        __LAUNCH_REVERSE_KERNEL__(512)
        __LAUNCH_REVERSE_KERNEL__(513)
    default:
        AT_ERROR("Unsupported channel dimension for reverse rasterize: ", CDIM);
    }
#undef __LAUNCH_REVERSE_KERNEL__

    return std::make_tuple(gauss_features, gauss_weights);
}



} // namespace gsplat