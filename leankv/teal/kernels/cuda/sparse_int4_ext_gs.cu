// sparse_int4_ext_gs.cu — graph-safe fused sparse-int4 GEMV torch extension.
//
// Fixes the CUDA-graph-capture bug in sparse_int4_ext.cu: that version used a
// per-call torch::zeros accumulator + atomicAdd, which is not graph-replay-safe
// (the accumulator isn't cleanly re-zeroed on replay). Here, split-K writes to
// SEPARATE partial rows (every element written exactly once, no accumulation),
// then a reduction pass sums them and writes fp16 directly. No atomics, no
// pre-zeroing, no .to() cast — every buffer element is written before read, so
// replay is deterministic regardless of buffer contents.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>   // launch on the current stream so CUDA graph capture records our kernels

#define VEC 8
#define G   128
#define SPLITK 8

// Kernel 1: each K-split computes a partial GEMV and WRITES its own row.
// yp : [SPLITK, N] fp32 ; x : [K] fp16 ; Wq : [K, N/2] u8 ; scales : [K/G, N] fp16
__global__ void int4_partial(float *__restrict__ yp, const half *__restrict__ x,
                             const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                             float t, int N, int K) {
    const int split = blockIdx.y;
    const int BLOCK_K = (K + SPLITK - 1) / SPLITK;
    const int k0 = split * BLOCK_K, k1 = min(k0 + BLOCK_K, K);
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cur_g = -1; float sc[VEC];

    for (int k = k0; k < k1; k++) {
        const float xk = __half2float(x[k]);
        if (fabsf(xk) <= t) continue;
        const int g = k / G;
        if (g != cur_g) {
            const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + nb]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int j = 0; j < VEC; j++) sc[j] = __half2float(sh[j]);
            cur_g = g;
        }
        const uint32_t w = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)k * Nh + (nb >> 1)]);
#pragma unroll
        for (int j = 0; j < VEC; j++)
            acc[j] += xk * (((int)((w >> (4 * j)) & 0xF) - 8) * sc[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC; j++)
        yp[(size_t)split * N + nb + j] = acc[j];      // direct write, no atomic
}

// Kernel 2: sum the SPLITK partials, write fp16 directly.
__global__ void reduce_fp16(half *__restrict__ y, const float *__restrict__ yp, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < SPLITK; i++) s += yp[(size_t)i * N + n];
    y[n] = __float2half(s);
}

torch::Tensor sparse_int4_gemv(torch::Tensor x, torch::Tensor Wq,
                               torch::Tensor scales, double threshold) {
    const int K = Wq.size(0);
    const int N = Wq.size(1) * 2;
    auto x2 = x.reshape({K}).contiguous();
    auto yp = torch::empty({SPLITK, N}, x.options().dtype(torch::kFloat32));
    auto y  = torch::empty({N}, x.options().dtype(torch::kHalf));

    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 b1(256), g1((N + 256 * VEC - 1) / (256 * VEC), SPLITK);
    int4_partial<<<g1, b1, 0, stream>>>(yp.data_ptr<float>(),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        (float)threshold, N, K);

    const int t2 = 256;
    reduce_fp16<<<(N + t2 - 1) / t2, t2, 0, stream>>>(
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), yp.data_ptr<float>(), N);

    return y.reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_int4_gemv", &sparse_int4_gemv, "graph-safe fused sparse-int4 GEMV (B=1)");
}
