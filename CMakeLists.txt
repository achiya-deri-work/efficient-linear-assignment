cmake_minimum_required(VERSION 3.20)
project(efficient_linear_assignment LANGUAGES CXX CUDA)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(Torch REQUIRED)

# Add source directory
add_subdirectory(src/csrc)
