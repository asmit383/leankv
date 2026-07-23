// sparse_int4_gemv_lop3.cu — LOP3 int4->fp16 unpack (Phase 3).
//
// Benchmarks the naive shift/mask unpack against the LOP3 bit-trick unpack, same
// fused sparse-int4 GEMV, same data. The LOP3 path converts 4-bit -> fp16 without
// an integer->float instruction: OR the nibble into an fp16 with exponent 0x6400
// (== 1024.0), so the bit pattern *is* 1024+n; then subtract 1032 (=1024+8) to get
// the signed value q = n-8. `lop3.b32 ...,0xEA` computes (a & MASK) | EX in ONE
// instruction, and mask 0x000f000f pulls two nibbles into a half2 at once.
//
// With natural packing (nibble j at bit 4j), the four lop3 ops emit values in the
// lane order {n0,n4},{n1,n5},{n2,n6},{n3,n7} — a fixed permutation handled below.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define VEC 8
#define G   128
#define CUDA_CHECK(call)                                                     \
    do { cudaError_t _e = (call);                                            \
         if (_e != cudaSuccess) { fprintf(stderr, "CUDA %s @ %s:%d\n",       \
             cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

__device__ __forceinline__ uint32_t lop3_ea(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xEA;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;   // (a & b) | c
}

// ── naive: shift/mask + int->float ──────────────────────────────────────
template <int BK>
__global__ void kern_naive(float *__restrict__ y, const half *__restrict__ x,
                           const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                           float t, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK, klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cg = -1; float sc[VEC];
    for (int k = 0; k < klen; k++) {
        const int kk = k0 + k; const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= t) continue;
        const int g = kk / G;
        if (g != cg) { const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + nb]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int j = 0; j < VEC; j++) sc[j] = __half2float(sh[j]); cg = g; }
        const uint32_t w = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)kk * Nh + (nb >> 1)]);
#pragma unroll
        for (int j = 0; j < VEC; j++) acc[j] += xk * (((int)((w >> (4 * j)) & 0xF) - 8) * sc[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&y[nb + j], acc[j]);
}

// ── LOP3: half2 bit-trick unpack ────────────────────────────────────────
template <int BK>
__global__ void kern_lop3(float *__restrict__ y, const half *__restrict__ x,
                          const uint8_t *__restrict__ Wq, const half *__restrict__ scales,
                          float t, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK, klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    const int Nh = N >> 1;
    const half2 bias = __half2half2(__float2half(1032.0f));   // 1024 + 8
    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;
    int cg = -1; half2 sc2[4];                                // sc2[p] = {scale[p], scale[p+4]}
    for (int k = 0; k < klen; k++) {
        const int kk = k0 + k; const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= t) continue;
        const int g = kk / G;
        if (g != cg) { const float4 s = *reinterpret_cast<const float4 *>(&scales[(size_t)g * N + nb]);
            const half *sh = reinterpret_cast<const half *>(&s);
#pragma unroll
            for (int p = 0; p < 4; p++) sc2[p] = __halves2half2(sh[p], sh[p + 4]); cg = g; }
        const uint32_t w = *reinterpret_cast<const uint32_t *>(&Wq[(size_t)kk * Nh + (nb >> 1)]);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const uint32_t hb = lop3_ea(w >> (4 * p), 0x000f000f, 0x64006400);  // {1024+n_p, 1024+n_{p+4}}
            const half2 q = __hsub2(*reinterpret_cast<const half2 *>(&hb), bias); // {q_p, q_{p+4}}
            const half2 dq = __hmul2(q, sc2[p]);                                 // dequant weights
            acc[p]     += xk * __low2float(dq);      // column nb+p
            acc[p + 4] += xk * __high2float(dq);     // column nb+p+4
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&y[nb + j], acc[j]);
}

int main(int argc, char **argv) {
    int N = 14336, K = 4096; float sparsity = 0.40f; int iters = 300;
    const int threads = 256, BK = 512;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparsity")) sparsity = atof(argv[++i]);
    }
    if (N % VEC || K % G) { fprintf(stderr, "need N%%8==0, K%%128==0\n"); return 1; }
    const float t = 0.5f; const int Nh = N >> 1, NG = K / G;

    std::mt19937 rng(0); std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<half> hx(K); std::vector<float> hW((size_t)K * N); int active = 0;
    for (int k = 0; k < K; k++) { float v = (u(rng) < sparsity) ? 0.f : (1.f + u(rng));
        hx[k] = __float2half(v); if (fabsf(v) > t) active++; }
    for (size_t i = 0; i < (size_t)K * N; i++) hW[i] = u(rng) * 0.02f - 0.01f;

    std::vector<uint8_t> hWq((size_t)K * Nh, 0); std::vector<half> hSc((size_t)NG * N);
    std::vector<float> hWdq((size_t)K * N);
    for (int n = 0; n < N; n++) for (int g = 0; g < NG; g++) {
        float ma = 1e-8f; for (int k = g * G; k < (g + 1) * G; k++) ma = std::max(ma, fabsf(hW[(size_t)k * N + n]));
        float sf = __half2float(__float2half(ma / 7.0f)); hSc[(size_t)g * N + n] = __float2half(ma / 7.0f);
        for (int k = g * G; k < (g + 1) * G; k++) { int q = std::max(-8, std::min(7, (int)lroundf(hW[(size_t)k * N + n] / sf)));
            hWdq[(size_t)k * N + n] = q * sf; hWq[(size_t)k * Nh + (n >> 1)] |= ((uint8_t)(q + 8) & 0xF) << (4 * (n & 1)); }
    }

    half *dx, *dSc; uint8_t *dWq; float *dy;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(half))); CUDA_CHECK(cudaMalloc(&dWq, (size_t)K * Nh));
    CUDA_CHECK(cudaMalloc(&dSc, (size_t)NG * N * sizeof(half))); CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWq, hWq.data(), (size_t)K * Nh, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dSc, hSc.data(), (size_t)NG * N * sizeof(half), cudaMemcpyHostToDevice));
    dim3 block(threads), grid((N + threads * VEC - 1) / (threads * VEC), (K + BK - 1) / BK);

    auto run = [&](int which) {
        CUDA_CHECK(cudaMemset(dy, 0, N * sizeof(float)));
        if (which == 0) kern_naive<BK><<<grid, block>>>(dy, dx, dWq, dSc, t, N, K);
        else            kern_lop3 <BK><<<grid, block>>>(dy, dx, dWq, dSc, t, N, K);
    };
    auto err = [&]() {
        std::vector<float> hy(N); CUDA_CHECK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
        double e = 0; for (int n = 0; n < std::min(N, 128); n++) { double r = 0;
            for (int k = 0; k < K; k++) { float xk = __half2float(hx[k]); if (fabsf(xk) > t) r += (double)xk * hWdq[(size_t)k * N + n]; }
            e = std::max(e, fabs(r - hy[n]) / (fabs(r) + 1e-6)); } return e;
    };
    auto bench = [&](int which) {
        run(which); CUDA_CHECK(cudaDeviceSynchronize()); double e = err();
        cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
        cudaEventRecord(a); for (int i = 0; i < iters; i++) run(which); cudaEventRecord(b); cudaEventSynchronize(b);
        float ms; cudaEventElapsedTime(&ms, a, b); ms /= iters; return std::make_pair((double)ms, e);
    };

    auto nv = bench(0); auto lp = bench(1);
    double wbytes = (double)active * N * 0.5 + (double)active / G * N * 2 + K * 2 + N * 4;
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device      : %s\n", prop.name);
    printf("N x K       : %d x %d   sparsity %.0f%%\n", N, K, 100.0 * (1 - (double)active / K));
    printf("naive unpack: %.4f ms   %.1f%% bw   rel err %.2e\n", nv.first, 100 * wbytes / (nv.first * 1e-3) / 1e9 / 300, nv.second);
    printf("LOP3  unpack: %.4f ms   %.1f%% bw   rel err %.2e\n", lp.first, 100 * wbytes / (lp.first * 1e-3) / 1e9 / 300, lp.second);
    printf("LOP3 speedup: %.3fx\n", nv.first / lp.first);
    return 0;
}
