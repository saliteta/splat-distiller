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
__global__ void reverse_rasterize_to_gaussians_2dgs_kernel(
    const uint32_t C,
    const uint32_t N,
    const uint32_t n_isects,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const uint32_t tile_size,
    const uint32_t image_width,
    const uint32_t image_height,

    const vec2        *__restrict__ means2d,
    const scalar_t    *__restrict__ ray_transforms,
    const scalar_t *__restrict__ opacities, // [C, N] or [nnz] // Gaussian
                                        // opacities that support per-view
                                        // values.
    const scalar_t    *__restrict__ rendered_colors, // [C,H,W,CDIM]
    const int32_t    *__restrict__ tile_offsets,
    const int32_t    *__restrict__ flatten_ids,
    scalar_t         *__restrict__ gauss_features,   // [nnz,CDIM]
    scalar_t         *__restrict__ gauss_weights     // [nnz]
) {
    // map blocks: x=cam, y=tile_row, z=tile_col
    auto block = cg::this_thread_block();
    int cam = block.group_index().x;
    int tile_row = block.group_index().y;
    int tile_col = block.group_index().z;
    int tile_id = tile_row * tile_width + tile_col;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    const uint32_t block_size = block.size();
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end = (tile_id == tile_width*tile_height-1 && cam==C-1)
                      ? n_isects
                      : tile_offsets[cam * tile_height * tile_width + tile_id + 1];
    int num_batches = (range_end - range_start + block_size - 1) / block_size;

    bool inside = (i < image_height && j < image_width);
     bool done = !inside;

    // Shared memory: load Gaussian IDs and parameters
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    // stores the concatination for projected primitive source (x, y) and
    // opacity alpha
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]

    // these are row vectors of the ray transformation matrices for the current
    // batch of gaussians
    vec3 *u_Ms_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *v_Ms_batch =
        reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]); // [block_size]
    vec3 *w_Ms_batch =
        reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]); // [block_size]


    int tr = block.thread_rank();

    // track transmittance per pixel (reset for each batch start)
    float T = 1.0f;

    // load Gaussians in batches and sum pixel contributions
    // iterate batches
    for (int b = 0; b < num_batches; ++b) {

        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        int batch_start = range_start + b * block_size;
        int idx = batch_start + tr;
        if (idx < range_end) {
            int32_t g = flatten_ids[idx];
            id_batch[tr] = g;
            vec2 m = means2d[g];
            float op = opacities[g];
            xy_opacity_batch[tr] = {m.x, m.y, op};
            u_Ms_batch[tr] = {
                ray_transforms[g * 9],
                ray_transforms[g * 9 + 1],
                ray_transforms[g * 9 + 2]
            };
            v_Ms_batch[tr] = {
                ray_transforms[g * 9 + 3],
                ray_transforms[g * 9 + 4],
                ray_transforms[g * 9 + 5]
            };
            w_Ms_batch[tr] ={
                ray_transforms[g * 9 + 6],
                ray_transforms[g * 9 + 7],
                ray_transforms[g * 9 + 8]
            };
        }

        // load Gaussian infos
        block.sync();



        int batch_size = min(block_size, range_end - batch_start);
        for (int t = 0; t < batch_size; ++t) {
            if (!inside) continue;


            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;

            const vec3 u_M = u_Ms_batch[t];
            const vec3 v_M = v_Ms_batch[t];
            const vec3 w_M = w_Ms_batch[t];

            // h_u and h_v are the homogeneous plane representations (they are
            // contravariant to the points on the primitive plane)
            const vec3 h_u = px * w_M - u_M;
            const vec3 h_v = py * w_M - v_M;


            const vec3 ray_cross = glm::cross(h_u, h_v);
            if (ray_cross.z == 0.0)
                continue;

            const vec2 s =
                vec2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            // IMPORTANT: This is where the gaussian kernel is evaluated!!!!!

            // point of interseciton in uv space
            const float gauss_weight_3d = s.x * s.x + s.y * s.y;

            // projected gaussian kernel
            const vec2 d = {xy_opac.x - px, xy_opac.y - py};
            // #define FILTER_INV_SQUARE_2DGS 2.0f
            const float gauss_weight_2d =
                FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);

            // merge ray-intersection kernel and 2d gaussian kernel
            const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

            const float sigma = 0.5f * gauss_weight;
            // evaluation of the gaussian exponential term
            float alpha = min(0.999f, opac * __expf(-sigma));

            // ignore transparent gaussians
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }



            // run volumetric rendering..
            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const scalar_t *pixel_color = rendered_colors + pix_id * CDIM;

#pragma unroll
            for (int k = 0; k < CDIM; ++k) {
                atomicAdd(&gauss_features[g*CDIM + k], pixel_color[k] * vis);
            }
            atomicAdd(&gauss_weights[g], vis);

            T = next_T;
        }
    }
}





    
    // Kernel launcher
    template <uint32_t CDIM>
    void launch_reverse_rasterize_to_gaussians_2dgs_kernel(
        const at::Tensor &means2d,         // [C, N, 2] or [nnz, 2]
        const at::Tensor &ray_transforms,  // [C, N, 3, 3] or [nnz, 3, 3]
        const at::Tensor &opacities,  // [C, N, 3, 3] or [nnz, 3, 3]
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
        auto stream = at::cuda::getCurrentCUDAStream();
    
        uint32_t C        = tile_offsets.size(0);
        uint32_t tile_h   = tile_offsets.size(1);
        uint32_t tile_w   = tile_offsets.size(2);
        uint32_t N        = packed ? 0 : means2d.size(1);
        uint32_t n_isects = flatten_ids.size(0);
    
        dim3 threads(tile_size, tile_size, 1);
        dim3 grid(C, tile_h, tile_w);
    
        // Shared memory: same layout as the kernel expects
        int64_t shmem_size = tile_size * tile_size * (
            sizeof(int32_t) +            // id_batch
            sizeof(vec3) * 4             // xy_opac_batch, uMs, vMs, wMs
        );
    
        // Allow large shared memory
        cudaFuncSetAttribute(
            reverse_rasterize_to_gaussians_2dgs_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        );
    
        // Launch the CUDA kernel
        reverse_rasterize_to_gaussians_2dgs_kernel<CDIM, float>
            <<<grid, threads, shmem_size, stream>>>(
                C,
                N,
                n_isects,
                tile_w,
                tile_h,
                tile_size,
                image_width,
                image_height,
                reinterpret_cast<vec2*>(means2d.data_ptr<float>()),
                ray_transforms.data_ptr<float>(),
                opacities.data_ptr<float>(),
                rendered_colors.data_ptr<float>(),
                tile_offsets.data_ptr<int32_t>(),
                flatten_ids.data_ptr<int32_t>(),
                gauss_features.data_ptr<float>(),
                gauss_weights.data_ptr<float>()
            );
    }
    
    // Explicit instantiations
    #define __REVERSE_INS__(CDIM)                                               \
        template void launch_reverse_rasterize_to_gaussians_2dgs_kernel<CDIM>(  \
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
}