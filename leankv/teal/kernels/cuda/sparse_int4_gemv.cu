// sparse_int4_gemv.cu — fused sparse + int4-dequant GEMV for batch-1 decode.
//
//   y[n] = Σ_k ( |x[k]| > t ?  x[k] · dequant_int4(Wq[k,n])  : 0 )
//
// Weights are int4 (symmetric, group-wise, G=128 along K), packed 2 nibbles/byte,
// column-major [K, N/2] so consecutive output columns stay contiguous. The int4
// weights stay packed in HBM; they are unpacked to fp in registers — never
// written back as fp16 — so we actually move 4x fewer weight bytes than fp16.
// Sparsity skips whole quantized rows for zeroed activations, on top of that.
//
// This version uses a naive shift/mask unpack (correctness first). The LOP3
// bit-trick unpack is the next optimization (see DESIGN_sparse_int4.md).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define VEC 8            // output columns per thread (one 32-bit packed load = 8 nibbles)
#define G   128          // quant group size along K
#define CUDA_CHECK(call)                                                     \
    do { cudaError_t _e = (call);                                            \
         if (_e != cudaSuccess) { fprintf(stderr, "CUDA %s @ %s:%d\n",       \
             cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

// y : [N] fp32 (zeroed) ; x : [K] fp16 ; Wq : [K, N/2] uint8 ; scales : [K/G, N] fp16
template <int BLOCK_K>
__global__ void sparse_int4_gemv_kernel(float *__restrict__ y,
                                        const half *__restrict__ x,
                                        const uint8_t *__restrict__ Wq,
                                        const half *__restrict__ scales,
                                        float threshold, int N, int K) {
    __shared__ half xs[BLOCK_K];
    const int k0 = blockIdx.y * BLOCK_K;
    const int klen = min(BLOCK_K, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();

    const int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (n_base >= N) return;
    const int Nh = N >> 1;                         // packed row width (bytes)

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;

    int cur_g = -1;
    float sc[VEC];                                 // per-column scales for current group

    for (int k = 0; k < klen; k++) {
        const int kk = k0 + k;
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= threshold) continue;      // skip whole quantized row

        const int g = kk / G;                      // reload scales only on group change
        if (g != cur_g) {
            const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + n_base]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int j = 0; j < VEC; j++) sc[j] = __half2float(sh[j]);
            cur_g = g;
        }

        // one 32-bit load = 8 nibbles = 8 output columns
        const uint32_t word = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)kk * Nh + (n_base >> 1)]);
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            const int q = (int)((word >> (4 * j)) & 0xF) - 8;   // offset-binary -> signed
            acc[j] += xk * (q * sc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++)
        atomicAdd(&y[n_base + j], acc[j]);
}

int main(int argc, char **argv) {
    int N = 14336, K = 4096;
    float sparsity = 0.40f;
    int threads = 256, iters = 200;
    const int BLOCK_K = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparsity")) sparsity = atof(argv[++i]);
    }
    if (N % VEC || K % G) { fprintf(stderr, "need N%%%d==0 and K%%%d==0\n", VEC, G); return 1; }
    const float threshold = 0.5f;
    const int Nh = N >> 1, NG = K / G;

    // ── host data + int4 group-wise quantization ────────────────────────
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<half> hx(K);
    std::vector<float> hW((size_t)K * N);          // reference fp weights, W[k*N+n]=weight[n,k]
    int active = 0;
    for (int k = 0; k < K; k++) {
        float v = (u(rng) < sparsity) ? 0.f : (1.f + u(rng));
        hx[k] = __float2half(v);
        if (fabsf(v) > threshold) active++;
    }
    for (size_t i = 0; i < (size_t)K * N; i++) hW[i] = u(rng) * 0.02f - 0.01f;

    std::vector<uint8_t> hWq((size_t)K * Nh, 0);
    std::vector<half> hSc((size_t)NG * N);
    std::vector<float> hWdq((size_t)K * N);        // dequantized (what the kernel effectively sees)
    for (int n = 0; n < N; n++) {
        for (int g = 0; g < NG; g++) {
            float maxabs = 1e-8f;
            for (int k = g * G; k < (g + 1) * G; k++) maxabs = std::max(maxabs, fabsf(hW[(size_t)k * N + n]));
            float scale = maxabs / 7.0f;
            hSc[(size_t)g * N + n] = __float2half(scale);
            float sf = __half2float(__float2half(scale));
            for (int k = g * G; k < (g + 1) * G; k++) {
                int q = (int)lroundf(hW[(size_t)k * N + n] / sf);
                q = std::max(-8, std::min(7, q));
                hWdq[(size_t)k * N + n] = q * sf;
                uint8_t nib = (uint8_t)(q + 8) & 0xF;              // offset binary
                hWq[(size_t)k * Nh + (n >> 1)] |= nib << (4 * (n & 1));
            }
        }
    }

    half *dx; uint8_t *dWq; half *dSc; float *dy;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dWq, (size_t)K * Nh));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t)NG * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), (size_t)K * Nh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, hSc.data(), (size_t)NG * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(threads), grid((N + threads * VEC - 1) / (threads * VEC), (K + BLOCK_K - 1) / BLOCK_K);
    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(dy, 0, N * sizeof(float)));
        sparse_int4_gemv_kernel<BLOCK_K><<<grid, block>>>(dy, dx, dWq, dSc, threshold, N, K);
    };
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());

    // correctness vs dequant-then-GEMV reference (same quantized weights) → tight
    std::vector<float> hy(N);
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    double rel = 0;
    for (int n = 0; n < std::min(N, 128); n++) {
        double ref = 0;
        for (int k = 0; k < K; k++) {
            float xk = __half2float(hx[k]);
            if (fabsf(xk) > threshold) ref += (double)xk * hWdq[(size_t)k * N + n];
        }
        rel = std::max(rel, fabs(ref - hy[n]) / (fabs(ref) + 1e-6));
    }

    cudaEvent_t a, b; CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));
    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters;

    double wbytes = (double)active * N * 0.5;                       // int4 weight rows
    double sbytes = (double)active / G * N * sizeof(half);          // scales touched
    double bytes = wbytes + sbytes + K * 2 + N * 4;
    double gbps = bytes / (ms * 1e-3) / 1e9;
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    printf("device            : %s\n", prop.name);
    printf("N x K             : %d x %d   sparsity %.0f%%  (%d/%d active)\n",
           N, K, 100.0 * (1 - (double)active / K), active, K);
    printf("weight bytes (int4): %.2f MB   (fp16 would be %.2f MB)\n", wbytes / 1e6, active * (double)N * 2 / 1e6);
    printf("time / call       : %.4f ms\n", ms);
    printf("effective bw      : %.1f GB/s  (%.1f%% of 300)\n", gbps, 100 * gbps / 300.0);
    printf("max rel err       : %.2e  (kernel vs dequant reference)\n", rel);
    return 0;
}
