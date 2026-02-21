#include <torch/extension.h>

// Declarations
std::vector<torch::Tensor> solve_auction_cuda(torch::Tensor cost_matrix, float epsilon, int max_iter);

std::vector<torch::Tensor> sinkhorn_cuda_forward(
    torch::Tensor C,
    torch::Tensor log_mu,
    torch::Tensor log_nu,
    float epsilon,
    int max_iter
);

std::vector<torch::Tensor> dual_ascent_cuda_forward(
    torch::Tensor C,
    torch::Tensor mu,
    torch::Tensor nu,
    float epsilon,
    int max_iter
);



// Declarations (Forward Refs)
std::vector<torch::Tensor> sinkhorn_cutlass_forward(
    torch::Tensor C,
    torch::Tensor log_mu,
    torch::Tensor log_nu,
    float epsilon,
    int max_iter
);
std::vector<torch::Tensor> dual_ascent_cutlass_forward(
    torch::Tensor C,
    torch::Tensor mu,
    torch::Tensor nu,
    float epsilon,
    int max_iter
);
std::vector<torch::Tensor> auction_cutlass_forward(
    torch::Tensor C,
    float epsilon,
    int max_iter
);

// Flash Forward Declarations
std::vector<torch::Tensor> flash_dual_ascent_dispatch(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter, std::string math_mode
);

std::vector<torch::Tensor> flash_dual_ascent_v3_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter
);

std::vector<torch::Tensor> flash_dual_ascent_v4_forward(
    torch::Tensor Q, torch::Tensor K, torch::Tensor mu, torch::Tensor nu,
    float epsilon, int max_iter
);

PYBIND11_MODULE(efficient_linear_assignment_cpp, m) {
    m.def("solve_auction_cuda", &solve_auction_cuda, "Solve Auction (CUDA)");
    m.def("sinkhorn_cuda_forward", &sinkhorn_cuda_forward, "Sinkhorn Persistent (CUDA)");
    m.def("dual_ascent_cuda_forward", &dual_ascent_cuda_forward, "Dual Ascent Persistent (CUDA)");
    
    m.def("sinkhorn_cutlass_forward", &sinkhorn_cutlass_forward, "Sinkhorn CUTLASS (CUDA)");
    m.def("dual_ascent_cutlass_forward", &dual_ascent_cutlass_forward, "Dual Ascent CUTLASS (CUDA)");
    m.def("auction_cutlass_forward", &auction_cutlass_forward, "Auction CUTLASS (Exact)");

    // Flash Kernels (V3/V4 consolidated)
    m.def("dual_ascent_implicit_v3_forward", &flash_dual_ascent_v3_forward, "Flash V3: SM80 Mixed Precision");
    m.def("dual_ascent_v4_forward", &flash_dual_ascent_v4_forward, "Flash V4: SM80 TF32 Optimized");
    
    // Generic Dispatch for Stability Testing
    m.def("flash_dual_ascent_dispatch", &flash_dual_ascent_dispatch, "Flash Generic: Control Math Mode");
}

// Declaration

