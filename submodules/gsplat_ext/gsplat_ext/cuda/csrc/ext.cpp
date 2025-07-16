#include "Ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rasterize_reverse_fwd_3dgs", &gsplat_ext::reverse_rasterize_to_gaussians_3dgs);
    m.def("rasterize_reverse_fwd_2dgs", &gsplat_ext::reverse_rasterize_to_gaussians_2dgs);
    m.def("rasterize_reverse_fwd_dbs", &gsplat_ext::reverse_rasterize_to_gaussians_dbs);
}
