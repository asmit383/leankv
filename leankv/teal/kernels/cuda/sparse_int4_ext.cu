// sparse_int4_ext.cu — torch extension for the fused sparse-int4 GEMV, so it can
// drive a real model's decode. Kernel identical to sparse_int4_gemv.cu.

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define VEC 8
#define G   128
#define BLOCK_K 512

template <int BK>
__global__ void sparse_int4_gemv_kernel(float *__restrict__ y,
                                        const half *__restrict__ x,
                                        const uint8_t *__restrict__ Wq,
                                        const half *__restrict__ scales,
                                        float threshold, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK, klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();

    const int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (n_base >= N) return;
    const int Nh = N >> 1;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cur_g = -1; float sc[VEC];

    for (int k = 0; k < klen; k++) {
        const int kk = k0 + k;
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= threshold) continue;
        const int g = kk / G;
        if (g != cur_g) {
            const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + n_base]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int j = 0; j < VEC; j++) sc[j] = __half2float(sh[j]);
            cur_g = g;
        }
        const uint32_t word = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)kk * Nh + (n_base >> 1)]);
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            const int q = (int)((word >> (4 * j)) & 0xF) - 8;
            acc[j] += xk * (q * sc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&y[n_base + j], acc[j]);
}

torch::Tensor sparse_int4_gemv(torch::Tensor x, torch::Tensor Wq,
                               torch::Tensor scales, double threshold) {
    const int K = Wq.size(0);
    const int N = Wq.size(1) * 2;
    auto x2 = x.reshape({K}).contiguous();
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));
    dim3 block(256), grid((N + 256 * VEC - 1) / (256 * VEC), (K + BLOCK_K - 1) / BLOCK_K);
    sparse_int4_gemv_kernel<BLOCK_K><<<grid, block>>>(
        y.data_ptr<float>(),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        Wq.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(scales.data_ptr<at::Half>()),
        (float)threshold, N, K);
    return y.to(torch::kHalf).reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_int4_gemv", &sparse_int4_gemv, "fused sparse-int4 GEMV (B=1)");
}
