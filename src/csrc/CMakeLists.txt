
# Define the extension module
python_add_library(efficient_linear_assignment_cpp MODULE
    interface.cpp
    kernels.cu
)

# Link against Torch and Python
target_link_libraries(efficient_linear_assignment_cpp PRIVATE ${TORCH_LIBRARIES})
set_target_properties(efficient_linear_assignment_cpp PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# Install the extension into the python package
install(TARGETS efficient_linear_assignment_cpp DESTINATION src/efficient_linear_assignment)
