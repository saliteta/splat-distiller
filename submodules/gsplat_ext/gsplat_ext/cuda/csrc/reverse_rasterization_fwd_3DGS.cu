#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat_ext {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

    template <uint32_t CDIM, typename scalar_t>
    __global__ void reverse_rasterize_to_gaussian_3dgs_kernel(
        const uint32_t C,
        const uint32_t N,
        const uint32_t n_isects,
        const uint32_t image_width,
        const uint32_t image_height,
        const uint32_t tile_size,
        const uint32_t tile_width,
        const uint32_t tile_height,

        const vec2 *__restrict__ means2d,         // [I, N, 2] or [nnz, 2]
        const vec3 *__restrict__ conics,          // [I, N, 3] or [nnz, 3]
        const scalar_t *__restrict__ opacities,   // [I, N] or [nnz]
        const scalar_t *__restrict__ rendered_colors,      // [C,H,W,CDIM]
        const int32_t *__restrict__ tile_offsets, // [I, tile_height, tile_width]
        const int32_t *__restrict__ flatten_ids,  // [n_isects]
        scalar_t *__restrict__ gauss_features, // // [nnz,CDIM]
        scalar_t *__restrict__ gauss_weights // [nnz]
    ) {
        // each thread draws one pixel, but also timeshares caching gaussians in a
        // shared tile

        auto block = cg::this_thread_block();
        int32_t cam = block.group_index().x;
        uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
        uint32_t j = block.group_index().z * tile_size + block.thread_index().x;
        int32_t tile_id =block.group_index().y * tile_width + block.group_index().z;


        float px = (float)j + 0.5f;
        float py = (float)i + 0.5f;
        int32_t pix_id = i * image_width + j;

        // return if out of bounds
        // keep not rasterizing threads around for reading data
        bool inside = (i < image_height && j < image_width);
        bool done = !inside;


        // have all threads in tile process the same gaussians in batches
        // first collect gaussians between range.x and range.y in batches
        // which gaussians to look through in this tile
        int32_t range_start = tile_offsets[tile_id];
        int32_t range_end =
            (cam == C - 1) && (tile_id == tile_width * tile_height - 1)
                ? n_isects
                : tile_offsets[tile_id + 1];
        const uint32_t block_size = block.size();
        uint32_t num_batches =
            (range_end - range_start + block_size - 1) / block_size;


        extern __shared__ int s[];
        int32_t *id_batch = (int32_t *)s; // [block_size]
        vec3 *xy_opacity_batch =
            reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
        vec3 *conic_batch =
            reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]

        // current visibility left to render
        // transmittance is gonna be used in the backward pass which requires a high
        // numerical precision so we use double for it. However double make bwd 1.5x
        // slower so we stick with float for now.
        float T = 1.0f;


        // collect and process batches of gaussians
        // each thread loads one gaussian at a time before rasterizing its
        // designated pixel
        uint32_t tr = block.thread_rank();

        for (uint32_t b = 0; b < num_batches; ++b) {
            // resync all threads before beginning next batch
            // end early if entire tile is done
            if (__syncthreads_count(done) >= block_size) {
                break;
            }

            // each thread fetch 1 gaussian from front to back
            // index of gaussian to load
            uint32_t batch_start = range_start + block_size * b;
            uint32_t idx = batch_start + tr;
            if (idx < range_end) {
                int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
                id_batch[tr] = g;
                const vec2 xy = means2d[g];
                const float opac = opacities[g];
                xy_opacity_batch[tr] = {xy.x, xy.y, opac};
                conic_batch[tr] = conics[g];
            }

            // wait for other threads to collect the gaussians in batch
            block.sync();

            // process gaussians in the current batch for this pixel
            uint32_t batch_size = min(block_size, range_end - batch_start);
            for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
                const vec3 conic = conic_batch[t];
                const vec3 xy_opac = xy_opacity_batch[t];
                const float opac = xy_opac.z;
                const vec2 delta = {xy_opac.x - px, xy_opac.y - py};
                const float sigma = 0.5f * (conic.x * delta.x * delta.x +
                                            conic.z * delta.y * delta.y) +
                                    conic.y * delta.x * delta.y;
                float alpha = min(0.999f, opac * __expf(-sigma));
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    continue;
                }

                const float next_T = T * (1.0f - alpha);
                if (next_T <= 1e-4f) { // this pixel is done: exclusive
                    done = true;
                    break;
                }

                int32_t g = id_batch[t];
                const float vis = alpha * T;
                const scalar_t *pixel_color = rendered_colors + pix_id * CDIM;
    #pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    atomicAdd(&gauss_features[g*CDIM + k], pixel_color[k] * vis *vis);
                }
                atomicAdd(&gauss_weights[g], vis * vis);

                T = next_T;
            }
        }
    }

    template <uint32_t CDIM>
    void launch_reverse_rasterize_to_gaussians_3dgs_kernel(
        // Gaussian parameters
        const at::Tensor &means2d,   // [..., N, 2] or [nnz, 2]
        const at::Tensor &conics,    // [..., N, 3] or [nnz, 3]
        const at::Tensor &opacities, // [..., N]  or [nnz]
        const at::Tensor &rendered_colors, // [C, H, W, CDIM]
        const at::Tensor &tile_offsets,    // [C, tile_h, tile_w]
        const at::Tensor &flatten_ids,     // [n_isects]
        at::Tensor       &gauss_features,  // [nnz, CDIM]
        at::Tensor       &gauss_weights,   // [nnz]
        uint32_t          image_width,
        uint32_t          image_height,
        uint32_t          tile_size
    ) {
        bool packed = means2d.dim() == 2;

        uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
        uint32_t C = tile_offsets.size(0);; // number of images
        uint32_t tile_height = tile_offsets.size(-2);
        uint32_t tile_width = tile_offsets.size(-1);
        uint32_t n_isects = flatten_ids.size(0);

        // Each block covers a tile on the image. In total there are
        // I * tile_height * tile_width blocks.
        dim3 threads = {tile_size, tile_size, 1};
        dim3 grid = {C, tile_height, tile_width};

        int64_t shmem_size =
            tile_size * tile_size * (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3));

        // TODO: an optimization can be done by passing the actual number of
        // channels into the kernel functions and avoid necessary global memory
        // writes. This requires moving the channel padding from python to C side.
        if (cudaFuncSetAttribute(
                reverse_rasterize_to_gaussian_3dgs_kernel<CDIM, float>,
                cudaFuncAttributeMaxDynamicSharedMemorySize,
                shmem_size
            ) != cudaSuccess) {
            AT_ERROR(
                "Failed to set maximum shared memory size (requested ",
                shmem_size,
                " bytes), try lowering tile_size."
            );
        }

        reverse_rasterize_to_gaussian_3dgs_kernel<CDIM, float>
            <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
                C,
                N,
                n_isects,
                image_width,
                image_height,
                tile_size,
                tile_width,
                tile_height,
                reinterpret_cast<vec2*>(means2d.data_ptr<float>()),
                reinterpret_cast<vec3 *>(conics.data_ptr<float>()),
                opacities.data_ptr<float>(),
                rendered_colors.data_ptr<float>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                gauss_features.data_ptr<float>(),
                gauss_weights.data_ptr<float>()
            );
        }
    
// Explicit Instantiation: this should match how it is being called in .cpp
#define __REVERSE_INS__(CDIM)                                          \
    template void launch_reverse_rasterize_to_gaussians_3dgs_kernel<CDIM>(           \
        const at::Tensor&,                                                    \
        const at::Tensor&,                                                    \
        const at::Tensor&,                                                    \
        const at::Tensor&,                                                    \
        const at::Tensor&,                                                    \
        const at::Tensor&,                                                    \
        at::Tensor&,                                                          \
        at::Tensor&,                                                          \
        uint32_t,                                                             \
        uint32_t,                                                             \
        uint32_t                                                              \
    );

__REVERSE_INS__(1)
__REVERSE_INS__(2)
__REVERSE_INS__(3)
__REVERSE_INS__(4)
__REVERSE_INS__(5)
__REVERSE_INS__(8)
__REVERSE_INS__(9)
__REVERSE_INS__(16)
__REVERSE_INS__(17)
__REVERSE_INS__(32)
__REVERSE_INS__(33)
__REVERSE_INS__(64)
__REVERSE_INS__(65)
__REVERSE_INS__(128)
__REVERSE_INS__(129)
__REVERSE_INS__(256)
__REVERSE_INS__(257)
__REVERSE_INS__(512)
__REVERSE_INS__(513)
#undef __REVERSE_INS__
} // namespace gsplat
