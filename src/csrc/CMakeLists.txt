
# Define the extension module
python_add_library(efficient_linear_assignment_cpp MODULE
    interface.cpp
    kernels.cu
    sinkhorn.cu
    dual_ascent.cu
    sinkhorn_cutlass.cu
    dual_ascent_cutlass.cu
    auction_cutlass.cu
)

# Link against Torch and Python
# Find specific python library from Torch
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")
target_link_libraries(efficient_linear_assignment_cpp PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
set_target_properties(efficient_linear_assignment_cpp PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
)

# Define the extension name for PyBind
target_compile_definitions(efficient_linear_assignment_cpp PRIVATE 
    TORCH_EXTENSION_NAME=efficient_linear_assignment_cpp
)

# CUTLASS Includes
target_include_directories(efficient_linear_assignment_cpp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/cutlass/tools/util/include
)

# Install the extension into the python package
install(TARGETS efficient_linear_assignment_cpp DESTINATION efficient_linear_assignment)
