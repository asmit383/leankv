// sparse_gemv_hip.cpp — batch-1 sparse GEMV for AMD CDNA (MI300X, gfx942).
//
// HIP port of the CUDA kernel (sparse_gemv.cu). Same algorithm, tuned for CDNA:
//   y[n] = Σ_k ( |x[k]| > threshold ? x[k] · W[k,n] : 0 )
// Weight column-major (W[k*N+n] == weight[n,k]) → coalesced output columns.
//
// CDNA notes:
//   * Wavefront = 64 lanes (vs 32 on NVIDIA). threads=256 → 4 wavefronts/block;
//     64 lanes × float4 (16 B) = 1024 B per memory transaction, good for HBM3.
//   * __syncthreads / LDS shared memory identical to CUDA.
//   * MI300X HBM3 peak ≈ 5.3 TB/s — reproduce the bandwidth-utilization %.
//   * Below this, Kog hand-writes raw CDNA ISA (buffer/global_load_dwordx4 +
//     s_waitcnt scheduling); this HIP kernel is the level directly above that.
//
// Build: hipcc -O3 --offload-arch=gfx942 sparse_gemv_hip.cpp -o sparse_gemv_hip
// Run:   ./sparse_gemv_hip            (sweeps configs, prints best + correctness)

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define VEC 8
#define HIP_CHECK(call)                                                       \
    do { hipError_t _e = (call);                                              \
         if (_e != hipSuccess) { fprintf(stderr, "HIP %s @ %s:%d\n",          \
             hipGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

template <int BLOCK_K>
__global__ void sparse_gemv_kernel(float *__restrict__ y,
                                   const __half *__restrict__ x,
                                   const __half *__restrict__ W,
                                   float threshold, int N, int K) {
    __shared__ __half xs[BLOCK_K];
    const int k0 = blockIdx.y * BLOCK_K;
    const int klen = min(BLOCK_K, K - k0);
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
        const float4 wv =
            *reinterpret_cast<const float4 *>(&W[(size_t)(k0 + k) * N + n_base]);
        const __half *wh = reinterpret_cast<const __half *>(&wv);
#pragma unroll
        for (int j = 0; j < VEC; j++)
            acc[j] += xk * __half2float(wh[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC; j++)
        atomicAdd(&y[n_base + j], acc[j]);
}

template <int BK>
static double bench(float *dy, const __half *dx, const __half *dW, float t,
                    int N, int K, int threads, int iters, int active,
                    double peak, double *out_relerr,
                    const std::vector<__half> &hx, const std::vector<__half> &hW) {
    dim3 block(threads);
    dim3 grid((N + threads * VEC - 1) / (threads * VEC), (K + BK - 1) / BK);
    auto launch = [&]() {
        HIP_CHECK(hipMemset(dy, 0, N * sizeof(float)));
        sparse_gemv_kernel<BK><<<grid, block>>>(dy, dx, dW, t, N, K);
    };
    launch();
    HIP_CHECK(hipDeviceSynchronize());

    if (out_relerr) {                          // correctness vs fp32 CPU ref (first 64 cols)
        std::vector<float> hy(N);
        HIP_CHECK(hipMemcpy(hy.data(), dy, N * sizeof(float), hipMemcpyDeviceToHost));
        double e = 0;
        for (int n = 0; n < std::min(N, 64); n++) {
            double ref = 0;
            for (int k = 0; k < K; k++) {
                float xk = __half2float(hx[k]);
                if (fabsf(xk) > t) ref += (double)xk * __half2float(hW[(size_t)k * N + n]);
            }
            e = std::max(e, fabs(ref - hy[n]) / (fabs(ref) + 1e-6));
        }
        *out_relerr = e;
    }

    hipEvent_t a, b; HIP_CHECK(hipEventCreate(&a)); HIP_CHECK(hipEventCreate(&b));
    HIP_CHECK(hipEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    HIP_CHECK(hipEventRecord(b)); HIP_CHECK(hipEventSynchronize(b));
    float ms; hipEventElapsedTime(&ms, a, b); ms /= iters;
    double bytes = (double)active * N * sizeof(__half) + K * 2 + N * 4;
    return bytes / (ms * 1e-3) / 1e9;          // GB/s
}

int main(int argc, char **argv) {
    int N = 16384, K = 16384;      // large matrix → asymptotic HBM bandwidth
    float sparsity = 0.0f;         // dense = pure bandwidth ceiling; try 0.4 for sparse
    int iters = 200;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparsity")) sparsity = atof(argv[++i]);
    }
    const float t = 0.5f;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<__half> hx(K), hW((size_t)K * N);
    int active = 0;
    for (int k = 0; k < K; k++) {
        float v = (u(rng) < sparsity) ? 0.f : (1.f + u(rng));
        hx[k] = __float2half(v);
        if (fabsf(v) > t) active++;
    }
    for (size_t i = 0; i < (size_t)K * N; i++) hW[i] = __float2half(u(rng) * 0.02f - 0.01f);

    __half *dx, *dW; float *dy;
    HIP_CHECK(hipMalloc(&dx, K * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dW, (size_t)K * N * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dy, N * sizeof(float)));
    HIP_CHECK(hipMemcpy(dx, hx.data(), K * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, hW.data(), (size_t)K * N * sizeof(__half), hipMemcpyHostToDevice));

    hipDeviceProp_t prop; HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    double peak = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8.0) / 1e9;
    if (peak < 1000.0 || peak > 8000.0) peak = 5300.0;  // fallback: MI300X HBM3 ≈ 5.3 TB/s


    printf("device         : %s\n", prop.name);
    printf("N x K          : %d x %d   sparsity %.0f%%  (%d/%d active)\n",
           N, K, 100.0 * (1 - (double)active / K), active, K);
    printf("peak HBM bw    : %.0f GB/s\n\n", peak);

    double relerr = 0, best = 0; int best_t = 0;
    for (int threads : {128, 256, 512}) {           // 2 / 4 / 8 wavefronts per block
        double re = 0;
        double gbps = bench<512>(dy, dx, dW, t, N, K, threads, iters, active, peak,
                                 threads == 128 ? &re : nullptr, hx, hW);
        if (threads == 128) relerr = re;
        printf("  threads=%-3d    %.1f GB/s   %.1f%% bw\n", threads, gbps, 100 * gbps / peak);
        if (gbps > best) { best = gbps; best_t = threads; }
    }
    printf("\nBEST           : %.1f GB/s = %.1f%% of peak  (threads=%d)\n",
           best, 100 * best / peak, best_t);
    printf("correctness    : max rel err %.2e (vs fp32 reference)\n", relerr);
    return 0;
}
