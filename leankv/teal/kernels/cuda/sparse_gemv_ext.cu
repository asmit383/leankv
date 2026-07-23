// sparse_gemv_ext.cu — torch extension wrapping the raw CUDA sparse GEMV so it
// can be benchmarked head-to-head with the Triton kernel on identical tensors.
//
// Semantics identical to leankv/teal/kernels/sparse_gemv.py:
//   y[n] = sum_k ( |x[k]| > threshold ? x[k] * W[k,n] : 0 )
// W is the column-major weight (flat buffer [K,N] row-major, W[k*N+n]==weight[n,k]).

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define VEC 16
#define BLOCK_K 512

template <int BK>
__global__ void sparse_gemv_kernel(float *__restrict__ y,
                                   const half *__restrict__ x,
                                   const half *__restrict__ W,
                                   float threshold, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK;
    const int klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();

    const int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (n_base >= N) return;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;

    for (int k = 0; k < klen; k++) {
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= threshold) continue;
        const float4 *wp =
            reinterpret_cast<const float4 *>(&W[(size_t)(k0 + k) * N + n_base]);
        const float4 wv0 = wp[0];
        const float4 wv1 = wp[1];
        const half *wh0 = reinterpret_cast<const half *>(&wv0);
        const half *wh1 = reinterpret_cast<const half *>(&wv1);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            acc[j]     += xk * __half2float(wh0[j]);
            acc[j + 8] += xk * __half2float(wh1[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&y[n_base + j], acc[j]);
}

torch::Tensor sparse_gemv(torch::Tensor x, torch::Tensor W, double threshold) {
    TORCH_CHECK(W.scalar_type() == torch::kHalf, "W must be fp16");
    const int N = W.size(0);
    const int K = W.size(1);
    TORCH_CHECK(N % VEC == 0, "N must be divisible by ", VEC);

    auto x2 = x.reshape({K}).contiguous();
    auto y = torch::zeros({N}, x.options().dtype(torch::kFloat32));

    const int threads = 256;
    dim3 block(threads);
    dim3 grid((N + threads * VEC - 1) / (threads * VEC), (K + BLOCK_K - 1) / BLOCK_K);

    sparse_gemv_kernel<BLOCK_K><<<grid, block>>>(
        y.data_ptr<float>(),
        reinterpret_cast<const half *>(x2.data_ptr<at::Half>()),
        reinterpret_cast<const half *>(W.data_ptr<at::Half>()),
        (float)threshold, N, K);

    return y.to(torch::kHalf).reshape({1, 1, N});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sparse_gemv", &sparse_gemv, "raw CUDA sparse GEMV (B=1)");
}
