#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_sh_fwd", &bsplat::compute_sh_fwd_tensor);
    m.def("compute_sh_bwd", &bsplat::compute_sh_bwd_tensor);

    m.def("compute_sb_fwd", &bsplat::compute_sb_fwd_tensor);
    m.def("compute_sb_bwd", &bsplat::compute_sb_bwd_tensor);

    m.def(
        "quat_scale_to_covar_preci_fwd",
        &bsplat::quat_scale_to_covar_preci_fwd_tensor
    );
    m.def(
        "quat_scale_to_covar_preci_bwd",
        &bsplat::quat_scale_to_covar_preci_bwd_tensor
    );

    m.def("proj_fwd", &bsplat::proj_fwd_tensor);
    m.def("proj_bwd", &bsplat::proj_bwd_tensor);

    m.def("world_to_cam_fwd", &bsplat::world_to_cam_fwd_tensor);
    m.def("world_to_cam_bwd", &bsplat::world_to_cam_bwd_tensor);

    m.def(
        "fully_fused_projection_fwd", &bsplat::fully_fused_projection_fwd_tensor
    );
    m.def(
        "fully_fused_projection_bwd", &bsplat::fully_fused_projection_bwd_tensor
    );

    m.def("isect_tiles", &bsplat::isect_tiles_tensor);
    m.def("isect_offset_encode", &bsplat::isect_offset_encode_tensor);

    m.def("rasterize_to_pixels_fwd", &bsplat::rasterize_to_pixels_fwd_tensor);
    m.def("rasterize_to_pixels_bwd", &bsplat::rasterize_to_pixels_bwd_tensor);

    m.def(
        "rasterize_to_indices_in_range",
        &bsplat::rasterize_to_indices_in_range_tensor
    );

    // packed version
    m.def(
        "fully_fused_projection_packed_fwd",
        &bsplat::fully_fused_projection_packed_fwd_tensor
    );
    m.def(
        "fully_fused_projection_packed_bwd",
        &bsplat::fully_fused_projection_packed_bwd_tensor
    );
}
