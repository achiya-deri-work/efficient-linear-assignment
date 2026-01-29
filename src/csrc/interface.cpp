#include <torch/extension.h>

// Declare external function from kernels.cu
std::vector<torch::Tensor> solve_auction_cuda(
    torch::Tensor cost_matrix,
    double epsilon,
    int max_iter,
    bool persistent_mode
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

// Wrapper for solve
std::vector<torch::Tensor> solve_wrapper(torch::Tensor cost, double epsilon, int max_iter, bool persistent_mode) {
    return solve_auction_cuda(cost, epsilon, max_iter, persistent_mode);
}

PYBIND11_MODULE(efficient_linear_assignment_cpp, m) {
    m.doc() = "Efficient Linear Assignment C++ Backend";
    m.def("compute_bids", &compute_bids, "Compute Bids (CUDA)");
    m.def("solve_auction_cuda", &solve_wrapper, "Solve Auction (Pure C++)",
          py::arg("cost"), py::arg("epsilon"), py::arg("max_iter"), py::arg("persistent_mode") = false);
}
