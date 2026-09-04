// CUDA explicit (FTCS) stencil for the same finite-volume operator as the CPU code.
//
// The coefficient arrays (aE, aN, aP, bfix, dt/mass) are built once on the host by
// moldcool::Operator and copied to the device, so the kernel is exactly the CPU update:
//     T_P^{n+1} = T_P + (dt / m_P) [ b_P - (aP T_P - aE_P T_E - aE_W T_W - aN_P T_N - aN_S T_S) ]
// Two kernels are timed: v1 reads every neighbour from global memory (L1/L2 absorb the reuse on
// Ampere); v2 stages a (BX+2) x (BY+2) tile of T in shared memory. The result is checked against
// the CPU stepper after the same number of steps. Throughput is reported in million cell updates
// per second (MLUPS). Structure of arrays, double buffering, no atomics.
//
// Build (WSL, user-space toolchain via micromamba, see README):
//   nvcc -O3 -std=c++17 -arch=sm_86 -Iinclude cuda/ftcs2d.cu -o moldcool_cuda_bench
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "moldcool/core.hpp"
#include "moldcool/operator.hpp"
#include "moldcool/stepper.hpp"

using namespace moldcool;

#define CUDA_CHECK(x)                                                                          \
    do {                                                                                       \
        cudaError_t e = (x);                                                                   \
        if (e != cudaSuccess) {                                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
            std::exit(1);                                                                      \
        }                                                                                      \
    } while (0)

constexpr int BX = 32, BY = 8;

__global__ void ftcs_v1(int nx, int ny, const std::uint8_t* __restrict__ mask, const double* __restrict__ aE,
                        const double* __restrict__ aN, const double* __restrict__ aP, const double* __restrict__ bfix,
                        const double* __restrict__ dt_m, const double* __restrict__ T, double* __restrict__ Tn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    const int P = j * nx + i;
    const double tp = T[P];
    if (mask[P] != 0) { Tn[P] = tp; return; }
    double s = aP[P] * tp;
    if (i + 1 < nx) s -= aE[P] * T[P + 1];
    if (i > 0)      s -= aE[P - 1] * T[P - 1];
    if (j + 1 < ny) s -= aN[P] * T[P + nx];
    if (j > 0)      s -= aN[P - nx] * T[P - nx];
    Tn[P] = tp + dt_m[P] * (bfix[P] - s);
}

__global__ void ftcs_v2(int nx, int ny, const std::uint8_t* __restrict__ mask, const double* __restrict__ aE,
                        const double* __restrict__ aN, const double* __restrict__ aP, const double* __restrict__ bfix,
                        const double* __restrict__ dt_m, const double* __restrict__ T, double* __restrict__ Tn) {
    __shared__ double tile[BY + 2][BX + 2];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int i = blockIdx.x * BX + tx;
    const int j = blockIdx.y * BY + ty;
    const bool in = (i < nx && j < ny);
    const int P = j * nx + i;
    // centre
    tile[ty + 1][tx + 1] = in ? T[P] : 0.0;
    // halo: edge threads fetch one neighbour each
    if (tx == 0)      tile[ty + 1][0]      = (in && i > 0)      ? T[P - 1]  : 0.0;
    if (tx == BX - 1) tile[ty + 1][BX + 1] = (in && i + 1 < nx) ? T[P + 1]  : 0.0;
    if (ty == 0)      tile[0][tx + 1]      = (in && j > 0)      ? T[P - nx] : 0.0;
    if (ty == BY - 1) tile[BY + 1][tx + 1] = (in && j + 1 < ny) ? T[P + nx] : 0.0;
    __syncthreads();
    if (!in) return;
    const double tp = tile[ty + 1][tx + 1];
    if (mask[P] != 0) { Tn[P] = tp; return; }
    double s = aP[P] * tp;
    if (i + 1 < nx) s -= aE[P] * tile[ty + 1][tx + 2];
    if (i > 0)      s -= aE[P - 1] * tile[ty + 1][tx];
    if (j + 1 < ny) s -= aN[P] * tile[ty + 2][tx + 1];
    if (j > 0)      s -= aN[P - nx] * tile[ty][tx + 1];
    Tn[P] = tp + dt_m[P] * (bfix[P] - s);
}

template <class T>
static T* to_device(const std::vector<T>& v) {
    T* d = nullptr;
    CUDA_CHECK(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2048;
    const int steps = argc > 2 ? std::atoi(argv[2]) : 200;
    auto prob = make_unit_square_dirichlet<double>(N, Materials<double>::Unit(), [](double, double) { return 1.0; });
    Operator<double> op(prob);
    const int nx = op.grid.nx, ny = op.grid.ny, n = op.grid.n();
    const double dt = 0.9 * op.explicit_dt_gershgorin();

    std::vector<std::uint8_t> mask(n);
    std::vector<double> aP(n), dt_m(n);
    for (int P = 0; P < n; ++P) {
        mask[P] = static_cast<std::uint8_t>(op.mask[P]);
        aP[P] = op.aP(P);
        dt_m[P] = (op.mask[P] == Cell::Unknown) ? dt / op.mass[P] : 0.0;
    }
    auto* d_mask = to_device(mask);
    auto* d_aE = to_device(op.aE);
    auto* d_aN = to_device(op.aN);
    auto* d_aP = to_device(aP);
    auto* d_b = to_device(op.bfix);
    auto* d_dtm = to_device(dt_m);
    double *d_T = to_device(prob.T0), *d_Tn = to_device(prob.T0);

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, 0));
    const dim3 block(BX, BY), grid((nx + BX - 1) / BX, (ny + BY - 1) / BY);
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    auto time_kernel = [&](int version, std::vector<double>& result) {
        CUDA_CHECK(cudaMemcpy(d_T, prob.T0.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        // warm-up
        if (version == 1) ftcs_v1<<<grid, block>>>(nx, ny, d_mask, d_aE, d_aN, d_aP, d_b, d_dtm, d_T, d_Tn);
        else              ftcs_v2<<<grid, block>>>(nx, ny, d_mask, d_aE, d_aN, d_aP, d_b, d_dtm, d_T, d_Tn);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(d_T, prob.T0.data(), n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(e0));
        for (int s = 0; s < steps; ++s) {
            if (version == 1) ftcs_v1<<<grid, block>>>(nx, ny, d_mask, d_aE, d_aN, d_aP, d_b, d_dtm, d_T, d_Tn);
            else              ftcs_v2<<<grid, block>>>(nx, ny, d_mask, d_aE, d_aN, d_aP, d_b, d_dtm, d_T, d_Tn);
            std::swap(d_T, d_Tn);
        }
        CUDA_CHECK(cudaEventRecord(e1));
        CUDA_CHECK(cudaEventSynchronize(e1));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        result.resize(n);
        CUDA_CHECK(cudaMemcpy(result.data(), d_T, n * sizeof(double), cudaMemcpyDeviceToHost));
        return double(op.n_unknown) * steps / (ms * 1e-3) / 1e6;
    };
    std::vector<double> T1, T2;
    const double mlups1 = time_kernel(1, T1);
    const double mlups2 = time_kernel(2, T2);

    // CPU reference (serial) with the identical operator and step count.
    ThetaStepper<double> st(op, 0.0, dt);
    std::vector<double> Tc = prob.T0;
    const auto c0 = std::chrono::steady_clock::now();
    for (int s = 0; s < steps; ++s) st.step(Tc, s * dt);
    const double cpu_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - c0).count();
    const double cpu_mlups = double(op.n_unknown) * steps / cpu_s / 1e6;
    double d1 = 0, d2 = 0, scale = 0;
    for (int P = 0; P < n; ++P) {
        d1 = std::max(d1, std::abs(T1[P] - Tc[P]));
        d2 = std::max(d2, std::abs(T2[P] - Tc[P]));
        scale = std::max(scale, std::abs(Tc[P]));
    }
    std::printf("{\"gpu\":\"%s\",\"N\":%d,\"steps\":%d,\"gpu_v1_global_mlups\":%.1f,\"gpu_v2_shared_mlups\":%.1f,"
                "\"cpu_serial_mlups\":%.1f,\"max_rel_diff_v1\":%.2e,\"max_rel_diff_v2\":%.2e}\n",
                props.name, N, steps, mlups1, mlups2, cpu_mlups, d1 / scale, d2 / scale);
    return (d1 / scale < 1e-10 && d2 / scale < 1e-10) ? 0 : 1;
}
