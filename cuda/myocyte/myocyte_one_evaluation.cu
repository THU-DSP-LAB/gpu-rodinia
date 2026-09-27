#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "define.c"
#include "kernel_ecc.cu"
#include "kernel_cam.cu"
#include "kernel.cu"
#include "../../openmp/myocyte/ecc.c"
#include "../../openmp/myocyte/cam.c"

namespace {

constexpr std::size_t kEquations = EQUATIONS;
constexpr std::size_t kParameters = PARAMETERS;
constexpr std::size_t kCommunications = 3;
constexpr float kMillimolarToMicromolar = 1000.0f;
constexpr float kAbsoluteTolerance = 1.0e-5f;
constexpr float kRelativeTolerance = 1.0e-3f;
constexpr std::array<int, kCommunications> kStateOffsets{46, 61, 76};
constexpr std::array<int, kCommunications> kParameterOffsets{0, 5, 10};
constexpr std::array<int, kCommunications> kCalciumInputs{35, 36, 37};

template <std::size_t N>
std::array<float, N> read_values(const char* path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    std::array<float, N> values{};
    for (auto& value : values) {
        if (!(input >> value))
            throw std::runtime_error(std::string("incomplete input ") + path);
    }
    return values;
}

void check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " +
                                 cudaGetErrorString(status));
}

template <std::size_t N>
void compare_values(const std::array<float, N>& expected,
                    const std::array<float, N>& actual, const char* name) {
    for (std::size_t index = 0; index < N; ++index) {
        const auto reference = expected[index];
        const auto observed = actual[index];
        if (std::isnan(reference) && std::isnan(observed)) continue;
        if (std::isinf(reference) && observed == reference) continue;
        const auto tolerance = kAbsoluteTolerance +
            kRelativeTolerance * std::fabs(reference);
        if (std::isfinite(reference) && std::isfinite(observed) &&
            std::fabs(reference - observed) <= tolerance) continue;
        throw std::runtime_error(std::string(name) + "[" +
                                 std::to_string(index) + "] differs: CPU=" +
                                 std::to_string(reference) + " CUDA=" +
                                 std::to_string(observed));
    }
}

void cpu_evaluate(const std::array<float, kEquations>& state,
                  const std::array<float, kParameters>& parameters,
                  std::array<float, kEquations>& derivative,
                  std::array<float, kCommunications>& communication) {
    auto mutable_state = state;
    auto mutable_parameters = parameters;
    ecc(0.0f, mutable_state.data(), 0, mutable_parameters.data(), 0,
        derivative.data());
    for (std::size_t index = 0; index < kCommunications; ++index) {
        communication[index] = cam(0.0f, mutable_state.data(),
            kStateOffsets[index], mutable_parameters.data(),
            kParameterOffsets[index], derivative.data(),
            state[kCalciumInputs[index]] * kMillimolarToMicromolar);
    }
}

}  // namespace

int main() {
    const auto state = read_values<kEquations>("../../data/myocyte/y.txt");
    const auto parameters = read_values<kParameters>("../../data/myocyte/params.txt");
    float* device_state = nullptr;
    float* device_derivative = nullptr;
    float* device_parameters = nullptr;
    float* device_communication = nullptr;
    check_cuda(cudaMalloc(&device_state, sizeof(state)), "allocate state");
    check_cuda(cudaMalloc(&device_derivative, sizeof(state)), "allocate derivative");
    check_cuda(cudaMalloc(&device_parameters, sizeof(parameters)), "allocate parameters");
    check_cuda(cudaMalloc(&device_communication,
                          kCommunications * sizeof(float)), "allocate communication");
    check_cuda(cudaMemcpy(device_state, state.data(), sizeof(state),
                          cudaMemcpyHostToDevice), "copy state");
    check_cuda(cudaMemcpy(device_parameters, parameters.data(), sizeof(parameters),
                          cudaMemcpyHostToDevice), "copy parameters");
    check_cuda(cudaMemset(device_derivative, 0, sizeof(state)), "clear derivative");
    check_cuda(cudaMemset(device_communication, 0, kCommunications * sizeof(float)),
               "clear communication");
    kernel<<<2, NUMBER_THREADS>>>(0, device_state, device_derivative,
                                  device_parameters, device_communication);
    check_cuda(cudaGetLastError(), "launch Myocyte kernel");
    check_cuda(cudaDeviceSynchronize(), "synchronize Myocyte kernel");
    std::array<float, kEquations> actual_derivative{};
    std::array<float, kCommunications> actual_communication{};
    check_cuda(cudaMemcpy(actual_derivative.data(), device_derivative,
                          sizeof(actual_derivative), cudaMemcpyDeviceToHost),
               "read derivative");
    check_cuda(cudaMemcpy(actual_communication.data(), device_communication,
                          sizeof(actual_communication), cudaMemcpyDeviceToHost),
               "read communication");
    std::array<float, kEquations> expected_derivative{};
    std::array<float, kCommunications> expected_communication{};
    cpu_evaluate(state, parameters, expected_derivative, expected_communication);
    compare_values(expected_derivative, actual_derivative, "derivative");
    compare_values(expected_communication, actual_communication, "communication");
    check_cuda(cudaFree(device_state), "free state");
    check_cuda(cudaFree(device_derivative), "free derivative");
    check_cuda(cudaFree(device_parameters), "free parameters");
    check_cuda(cudaFree(device_communication), "free communication");
    std::cout << "MYOCYTE_ONE_EVALUATION_PASS equations=" << kEquations
              << " communications=" << kCommunications << '\n';
}
