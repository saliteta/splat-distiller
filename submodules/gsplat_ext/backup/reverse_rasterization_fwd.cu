#include "Ops.h"
#include "helpers.cuh"
#include "types.cuh"
#include "Common.h"
#include <cooperative_groups.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <tuple>

namespace gsplat_ext {

namespace cg = cooperative_groups;

/****************************************************************************
 * Rasterization to Pixels Forward Pass
 ****************************************************************************/



template <uint32_t COLOR_DIM, typename S>
__global__ void rasterize_reverse_fwd_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const bool packed,
    const vec2<S> *__restrict__ means2d,  // [C, N, 2] or [nnz, 2]
    const vec3<S> *__restrict__ conics,   // [C, N, 3] or [nnz, 3]
    const S *__restrict__ opacities,      // [C, N] or [nnz]
    const S *__restrict__ input_image,    // [C, H, W, COLOR_DIM]
    const bool *__restrict__ masks,       // [C, tile_height, tile_width]
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [C, tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]
    S *__restrict__ gaussian_features,          // [C, N, COLOR_DIM] or [nnz, COLOR_DIM]
    S *__restrict__ weight      // [C, N] or [nnz] (optional)
) {
    auto block = cg::this_thread_block();
    int32_t camera_id = block.group_index().x;
    int32_t tile_id = block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += camera_id * tile_height * tile_width;
    input_image += camera_id * image_height * image_width * COLOR_DIM;
    if (masks != nullptr) {
        masks += camera_id * tile_height * tile_width;
    }

    S px = (S)j + 0.5f;
    S py = (S)i + 0.5f;
    int32_t pix_id = i * image_width + j;
    bool inside = (i < image_height && j < image_width);

    // Skip masked tiles
    if (masks != nullptr && inside && !masks[tile_id]) {
        return;
    }

    // Shared memory for batch processing
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s;
    vec3<S> *xy_opacity_batch = reinterpret_cast<vec3<S>*>(&id_batch[block.size()]);
    vec3<S> *conic_batch = reinterpret_cast<vec3<S>*>(&xy_opacity_batch[block.size()]);

    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == tile_width * tile_height - 1) ? 
                        n_isects : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches = (range_end - range_start + block_size - 1) / block_size;

    S T = 1.0f;
    for (uint32_t b = 0; b < num_batches; ++b) {
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + block.thread_rank();
        
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[block.thread_rank()] = g;
            vec2<S> mean2d = means2d[g];
            xy_opacity_batch[block.thread_rank()] = {
                mean2d.x, mean2d.y, opacities[g]
            };
            conic_batch[block.thread_rank()] = conics[g];
        }
        block.sync();

        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; t < batch_size; ++t) {
            if (!inside) continue;

            vec3<S> conic = conic_batch[t];
            vec3<S> xy_opac = xy_opacity_batch[t];
            S opacity = xy_opac.z;
            
            // Calculate Gaussian contribution
            vec2<S> delta = {xy_opac.x - px, xy_opac.y - py};
            S sigma = 0.5f * (conic.x * delta.x * delta.x + 
                            conic.z * delta.y * delta.y) + 
                     conic.y * delta.x * delta.y;
            
            S alpha = min(0.999f, opacity * __expf(-sigma));
            if (sigma < 0.f || alpha < 1.f/255.f) continue;

            S vis = alpha * T;
            T *= (1.0f - alpha);

            // Get input pixel color
            const S* pixel_color = input_image + pix_id * COLOR_DIM;
            int32_t g = id_batch[t];

            // Atomic add to Gaussian's color buffer
            for (uint32_t k = 0; k < COLOR_DIM; ++k) {
                atomicAdd(&gaussian_features[g * COLOR_DIM + k], pixel_color[k] * vis);
            }
            atomicAdd(&weight[g], vis);
        }
        block.sync();
    }
}





template <uint32_t CDIM>
std::tuple<torch::Tensor, torch::Tensor> call_kernel_with_dim(
    // Gaussian parameters
    const torch::Tensor &means2d,   // [C, N, 2] or [nnz, 2]
    const torch::Tensor &conics,    // [C, N, 3] or [nnz, 3]
    const torch::Tensor &padded_images,    // [C, H, W, Feature Number]
    const torch::Tensor &opacities, // [C, N]  or [nnz]
    const at::optional<torch::Tensor> &masks, // [C, tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const torch::Tensor &tile_offsets, // [C, tile_height, tile_width]
    const torch::Tensor &flatten_ids   // [n_isects]
) {
    GSPLAT_DEVICE_GUARD(means2d);
    GSPLAT_CHECK_INPUT(means2d);
    GSPLAT_CHECK_INPUT(conics);
    GSPLAT_CHECK_INPUT(padded_images);
    GSPLAT_CHECK_INPUT(opacities);
    GSPLAT_CHECK_INPUT(tile_offsets);
    GSPLAT_CHECK_INPUT(flatten_ids);
    if (masks.has_value()) {
        GSPLAT_CHECK_INPUT(masks.value());
    }
    bool packed = means2d.dim() == 2;

    uint32_t C = tile_offsets.size(0);         // number of cameras
    uint32_t N = packed ? 0 : means2d.size(1); // number of gaussians
    uint32_t channels = padded_images.size(-1);
    uint32_t tile_height = tile_offsets.size(1);
    uint32_t tile_width = tile_offsets.size(2);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // C * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 blocks = {C, tile_height, tile_width};

    torch::Tensor feature_gaussian = torch::zeros(
        {means2d.size(0), channels},
        means2d.options().dtype(torch::kFloat32)
    );
    torch::Tensor feature_weight = torch::zeros(
        {means2d.size(0)},
        means2d.options().dtype(torch::kFloat32)
    );

    at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
    const uint32_t shared_mem =
        tile_size * tile_size *
        (sizeof(int32_t) + sizeof(vec3<float>) + sizeof(vec3<float>));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_reverse_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shared_mem,
            " bytes), try lowering tile_size."
        );
    }
    rasterize_reverse_fwd_kernel<CDIM, float>
        <<<blocks, threads, shared_mem, stream>>>(
            C,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2<float> *>(means2d.data_ptr<float>()),
            reinterpret_cast<vec3<float> *>(conics.data_ptr<float>()),
            opacities.data_ptr<float>(),
            padded_images.data_ptr<float>(),
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            feature_gaussian.data_ptr<float>(),
            feature_weight.data_ptr<float>()
        );

    return std::make_tuple(feature_gaussian, feature_weight);
}

std::tuple<torch::Tensor, torch::Tensor> rasterize_reverse_fwd_tensor(
    const torch::Tensor &means2d,
    const torch::Tensor &conics,
    const torch::Tensor &opacities,
    const torch::Tensor &input_image,
    const at::optional<torch::Tensor> &masks,
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const torch::Tensor &tile_offsets,
    const torch::Tensor &flatten_ids
) {
    GSPLAT_CHECK_INPUT(input_image);
    uint32_t channels = input_image.size(-1);
    

    #define __GS__CALL_(N)                                                         \
        case N:                                                                    \
            return call_kernel_with_dim<N>(                                        \
                means2d,                                                           \
                conics,                                                            \
                input_image,                                                            \
                opacities,                                                         \
                masks,                                                             \
                image_width,                                                       \
                image_height,                                                      \
                tile_size,                                                         \
                tile_offsets,                                                      \
                flatten_ids                                                        \
            );

    // Dispatch based on padded channel count
    switch (channels) {
        __GS__CALL_(1)
        __GS__CALL_(2)
        __GS__CALL_(3)
        __GS__CALL_(4)
        __GS__CALL_(5)
        __GS__CALL_(8)
        __GS__CALL_(9)
        __GS__CALL_(16)
        __GS__CALL_(17)
        __GS__CALL_(32)
        __GS__CALL_(33)
        __GS__CALL_(64)
        __GS__CALL_(65)
        __GS__CALL_(128)
        __GS__CALL_(129)
        __GS__CALL_(256)
        __GS__CALL_(257)
        __GS__CALL_(512)
        __GS__CALL_(513)
        default:
            AT_ERROR("Unsupported padded channel count: ", channels);
    }
}
} // namespace gsplat