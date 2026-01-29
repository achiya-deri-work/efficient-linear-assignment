
# Define the extension module
python_add_library(efficient_linear_assignment_cpp MODULE
    interface.cpp
    kernels.cu
)

# Link against Torch and Python
# Find specific python library from Torch
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(efficient_linear_assignment_cpp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(efficient_linear_assignment_cpp PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# Install the extension into the python package
install(TARGETS efficient_linear_assignment_cpp DESTINATION efficient_linear_assignment)
