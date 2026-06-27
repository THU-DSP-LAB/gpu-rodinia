#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

inline void check_cuda_result(cudaError_t result, const char* expr, const char* file, int line) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s:%d: %s returned %s\n", file, line, expr, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(expr) check_cuda_result((expr), #expr, __FILE__, __LINE__)

inline void getLastCudaError(const char* message) {
    cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", message, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}
