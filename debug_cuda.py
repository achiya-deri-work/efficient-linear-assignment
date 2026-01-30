import sys
# sys.path.append('src')

try:
    from efficient_linear_assignment.auction.cpp_backend import efficient_linear_assignment_cpp
    print("Extension loaded:", efficient_linear_assignment_cpp)
    print("Methods:", dir(efficient_linear_assignment_cpp))
except ImportError as e:
    print("ImportError:", e)
    
try:
    from efficient_linear_assignment.sinkhorn import cuda_backend
    print("Sinkhorn CUDA Module:", cuda_backend)
except Exception as e:
    print("Sinkhorn CUDA Error:", e)

try:
    from efficient_linear_assignment.sinkhorn.api import BACKENDS
    print("Registered Sinkhorn Backends:", BACKENDS.keys())
except Exception as e:
    print("Sinkhorn API Error:", e)
