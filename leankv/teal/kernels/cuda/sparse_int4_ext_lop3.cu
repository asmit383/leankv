// sparse_int4_ext_lop3.cu — fast fused-int4 GEMV ext with LOP3 half2 unpack.
// Split-K + atomicAdd (fast, eager — not graph-safe, and we don't need graphs),
// LOP3 bit-trick int4->fp16 unpack, launched on the current stream.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

#define VEC 8
#define G   128
#ifndef BLOCK_K
#define BLOCK_K 512      // override with -DBLOCK_K=N to tune split-K
#endif

__device__ __forceinline__ uint32_t lop3_ea(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d; asm("lop3.b32 %0,%1,%2,%3,0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c)); return d;
}

__global__ void int4_lop3(float *__restrict__ y, const half *__restrict__ x,
                          const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                          float t, int N, int K) {
    __shared__ half xs[BLOCK_K];
    const int k0 = blockIdx.y * BLOCK_K, klen = min(BLOCK_K, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;
    const half2 bias = __half2half2(__float2half(1032.0f));

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cg = -1; half2 sc2[4];

    for (int k = 0; k < klen; k++) {
        const int kk = k0 + k; const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= t) continue;
        const int g = kk / G;
        if (g != cg) {
            const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + nb]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int p = 0; p < 4; p++) sc2[p] = __halves2half2(sh[p], sh[p + 4]);
            cg = g;
        }
        const uint32_t w = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)kk * Nh + (nb >> 1)]);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t hb = lop3_ea(w >> (4 * p), 0x000f000f, 0x64006400);
            const half2 q = __hsub2(*reinterpret_cast<const half2 *>(&hb), bias);
            const half2 dq = __hmul2(q, sc2[p]);
            acc[p]     += xk * __low2float(dq);
            acc[p + 4] += xk * __high2float(dq);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&y[nb + j], acc[j]);
}

torch::Tensor sparse_int4_gemv(torch::Tensor x, torch::Tensor Wq,
                               torch::Tensor scales, double threshold) {
    const int K = Wq.size(0), N = Wq.size(1) * 2;
    auto x2 = x.reshape({K}).contiguous();
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 b(256), g((N + 256 * VEC - 1) / (256 * VEC), (K + BLOCK_K - 1) / BLOCK_K);
    int4_lop3<<<g, b, 0, stream>>>(y.data_ptr<float>(),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        (float)threshold, N, K);
    return y.to(torch::kHalf).reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_int4_gemv", &sparse_int4_gemv, "fused-int4 GEMV, LOP3 unpack");
}
