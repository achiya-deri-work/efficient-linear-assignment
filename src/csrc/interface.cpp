#include <torch/extension.h>

void init_kernels(py::module& m);

PYBIND11_MODULE(efficient_linear_assignment_cpp, m) {
    m.doc() = "Efficient Linear Assignment C++ Backend";
    // init_kernels(m); // Will uncomment when kernels are ready
}
