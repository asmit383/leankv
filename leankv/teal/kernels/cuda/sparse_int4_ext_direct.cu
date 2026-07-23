// sparse_int4_ext_direct.cu — single-pass fused-int4 GEMV, writes fp16 directly.
// No split-K, no atomics, no torch.zeros, no .to() — one thread owns VEC columns
// and loops over all K, writing the fp16 result. Minimal per-call overhead.
// Trade-off: lower occupancy for small-N/large-K projections (down_proj).

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define VEC 8
#define G   128

__global__ void int4_direct(half *__restrict__ y, const half *__restrict__ x,
                            const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                            float t, int N, int K) {
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cg = -1; float sc[VEC];

    for (int k = 0; k < K; k++) {
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
    for (int j = 0; j < VEC; j++) y[nb + j] = __float2half(acc[j]);
}

torch::Tensor sparse_int4_gemv(torch::Tensor x, torch::Tensor Wq,
                               torch::Tensor scales, double threshold) {
    const int K = Wq.size(0), N = Wq.size(1) * 2;
    auto x2 = x.reshape({K}).contiguous();
    auto y = torch::empty({N}, x.options().dtype(torch::kHalf));
    auto stream = at::cuda::getCurrentCUDAStream();
    const int threads = 128;
    int4_direct<<<(N + threads * VEC - 1) / (threads * VEC), threads, 0, stream>>>(
        reinterpret_cast<half *>(y.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        (float)threshold, N, K);
    return y.reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_int4_gemv", &sparse_int4_gemv, "single-pass fused-int4 GEMV, direct fp16");
}
