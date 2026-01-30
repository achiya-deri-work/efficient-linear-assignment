#include <torch/extension.h>

// Declarations
std::vector<torch::Tensor> solve_auction_cuda(torch::Tensor cost_matrix, float epsilon, int max_iter, bool use_persistent);

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

void launch_bid_kernel_cuda(
    torch::Tensor benefits,
    torch::Tensor prices,
    torch::Tensor assignment,
    torch::Tensor best_idx,
    torch::Tensor increments,
    double epsilon
);

void compute_bids(
    torch::Tensor benefits,
    torch::Tensor prices,
    torch::Tensor assignment,
    torch::Tensor best_idx,
    torch::Tensor increments,
    double epsilon
) {
    launch_bid_kernel_cuda(benefits, prices, assignment, best_idx, increments, epsilon);
}

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

PYBIND11_MODULE(efficient_linear_assignment_cpp, m) {
    m.def("solve_auction_cuda", &solve_auction_cuda, "Solve Auction (CUDA)");
    m.def("compute_bids", &compute_bids, "Compute Bids (CUDA)");
    m.def("sinkhorn_cuda_forward", &sinkhorn_cuda_forward, "Sinkhorn Persistent (CUDA)");
    m.def("dual_ascent_cuda_forward", &dual_ascent_cuda_forward, "Dual Ascent Persistent (CUDA)");
    
    m.def("sinkhorn_cutlass_forward", &sinkhorn_cutlass_forward, "Sinkhorn CUTLASS (CUDA)");
    m.def("dual_ascent_cutlass_forward", &dual_ascent_cutlass_forward, "Dual Ascent CUTLASS (CUDA)");
    m.def("auction_cutlass_forward", &auction_cutlass_forward, "Auction CUTLASS (Exact)");
}
