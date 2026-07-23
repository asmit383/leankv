// sparse_int4_ext_gs.cu — graph-safe fused-int4 GEMV, tunable split-K.
//
// Graph-safe (for CUDA graph capture): split-K writes SEPARATE partial rows
// (every element written once, no atomics, no pre-zeroing), then a reduction
// writes fp16 directly. Kernels launch on the current stream so capture records
// them. BLOCK_K tile is tunable (-DBLOCK_K=N) → nsplit = ceil(K/BLOCK_K).

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define VEC 8
#define G   128
#ifndef BLOCK_K
#define BLOCK_K 128
#endif

__global__ void int4_partial(float *__restrict__ yp, const half *__restrict__ x,
                             const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                             float t, int N, int K) {
    const int split = blockIdx.y;
    const int k0 = split * BLOCK_K, k1 = min(k0 + BLOCK_K, K);
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cg = -1; float sc[VEC];
    for (int k = k0; k < k1; k++) {
        const float xk = __half2float(x[k]);
        if (fabsf(xk) <= t) continue;
        const int g = k / G;
        if (g != cg) {
            const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + nb]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int j = 0; j < VEC; j++) sc[j] = __half2float(sh[j]);
            cg = g;
        }
        const uint32_t w = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)k * Nh + (nb >> 1)]);
#pragma unroll
        for (int j = 0; j < VEC; j++)
            acc[j] += xk * (((int)((w >> (4 * j)) & 0xF) - 8) * sc[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) yp[(size_t)split * N + nb + j] = acc[j];
}

__global__ void reduce_fp16(half *__restrict__ y, const float *__restrict__ yp, int N, int nsplit) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.f;
    for (int i = 0; i < nsplit; i++) s += yp[(size_t)i * N + n];
    y[n] = __float2half(s);
}

torch::Tensor sparse_int4_gemv(torch::Tensor x, torch::Tensor Wq,
                               torch::Tensor scales, double threshold) {
    const int K = Wq.size(0), N = Wq.size(1) * 2;
    const int nsplit = (K + BLOCK_K - 1) / BLOCK_K;
    auto x2 = x.reshape({K}).contiguous();
    auto yp = torch::empty({nsplit, N}, x.options().dtype(torch::kFloat32));
    auto y  = torch::empty({N}, x.options().dtype(torch::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 b1(256), g1((N + 256 * VEC - 1) / (256 * VEC), nsplit);
    int4_partial<<<g1, b1, 0, stream>>>(yp.data_ptr<float>(),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        (float)threshold, N, K);
    const int t2 = 256;
    reduce_fp16<<<(N + t2 - 1) / t2, t2, 0, stream>>>(
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), yp.data_ptr<float>(), N, nsplit);
    return y.reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_int4_gemv", &sparse_int4_gemv, "graph-safe fused-int4 GEMV, tunable split-K");
}
