// fused_gate_up.cu — fused sparse SwiGLU projection for B=1 decode.
//
// Llama/Mistral MLP does:  h = silu(gate_proj(x)) * up_proj(x)
// gate_proj and up_proj read the SAME input x with the SAME activation-sparsity
// mask. The unfused path runs two separate sparse matmuls and round-trips gate
// and up through HBM before the elementwise. This kernel fuses them: one pass
// loads x / computes the mask once and streams BOTH weight matrices, so the
// gate and up columns for a given active row k are produced together and the
// SiLU*mul is applied in a single cheap epilogue.
//
// Weight layout matches the Triton kernel: column-major, W[k*N+n] == weight[n,k].
// The fused weight is conceptually [gate|up] stacked → one matmul, two outputs.

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
#define BLOCK_K 512
#define CUDA_CHECK(call)                                                     \
    do { cudaError_t _e = (call);                                            \
         if (_e != cudaSuccess) { fprintf(stderr, "CUDA %s @ %s:%d\n",       \
             cudaGetErrorString(_e), __FILE__, __LINE__); exit(1);} } while (0)

// ── Unfused building block: one sparse matmul into an fp32 accumulator ──
template <int BK>
__global__ void sparse_gemv_kernel(float *__restrict__ acc,
                                   const half *__restrict__ x,
                                   const half *__restrict__ W,
                                   float t, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK, klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    float a[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) a[j] = 0.f;
    for (int k = 0; k < klen; k++) {
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= t) continue;
        const float4 w = *reinterpret_cast<const float4 *>(&W[(size_t)(k0 + k) * N + nb]);
        const half *wh = reinterpret_cast<const half *>(&w);
#pragma unroll
        for (int j = 0; j < VEC; j++) a[j] += xk * __half2float(wh[j]);
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) atomicAdd(&acc[nb + j], a[j]);
}

// ── Fused: stream gate + up weights together, one mask, one x pass ──
template <int BK>
__global__ void fused_gate_up_kernel(float *__restrict__ g_acc,
                                     float *__restrict__ u_acc,
                                     const half *__restrict__ x,
                                     const half *__restrict__ Wg,
                                     const half *__restrict__ Wu,
                                     float t, int N, int K) {
    __shared__ half xs[BK];
    const int k0 = blockIdx.y * BK, klen = min(BK, K - k0);
    for (int i = threadIdx.x; i < klen; i += blockDim.x) xs[i] = x[k0 + i];
    __syncthreads();
    const int nb = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (nb >= N) return;
    float ga[VEC], ua[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) { ga[j] = 0.f; ua[j] = 0.f; }
    for (int k = 0; k < klen; k++) {
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= t) continue;            // one mask decision, reused for both
        const float4 wg = *reinterpret_cast<const float4 *>(&Wg[(size_t)(k0 + k) * N + nb]);
        const float4 wu = *reinterpret_cast<const float4 *>(&Wu[(size_t)(k0 + k) * N + nb]);
        const half *pg = reinterpret_cast<const half *>(&wg);
        const half *pu = reinterpret_cast<const half *>(&wu);
#pragma unroll
        for (int j = 0; j < VEC; j++) {
            ga[j] += xk * __half2float(pg[j]);
            ua[j] += xk * __half2float(pu[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < VEC; j++) {
        atomicAdd(&g_acc[nb + j], ga[j]);
        atomicAdd(&u_acc[nb + j], ua[j]);
    }
}

// ── Epilogue: h = silu(gate) * up  (applied after split-K reduction) ──
__global__ void silu_mul_kernel(half *__restrict__ h, const float *__restrict__ g,
                                const float *__restrict__ u, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    const float gv = g[n];
    h[n] = __float2half((gv / (1.f + expf(-gv))) * u[n]);
}

static double ev_ms(cudaEvent_t a, cudaEvent_t b) { float m; cudaEventElapsedTime(&m, a, b); return m; }

int main(int argc, char **argv) {
    int N = 14336, K = 4096;          // Mistral-7B MLP: gate/up are 14336 x 4096
    float sparsity = 0.40f;
    int threads = 256, iters = 300;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparsity")) sparsity = atof(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }
    const float t = 0.5f;

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<half> hx(K), hWg((size_t)K * N), hWu((size_t)K * N);
    int active = 0;
    for (int k = 0; k < K; k++) {
        float v = (u(rng) < sparsity) ? 0.f : (1.f + u(rng));
        hx[k] = __float2half(v);
        if (fabsf(v) > t) active++;
    }
    for (size_t i = 0; i < (size_t)K * N; i++) {
        hWg[i] = __float2half(u(rng) * 0.02f - 0.01f);
        hWu[i] = __float2half(u(rng) * 0.02f - 0.01f);
    }

    half *dx, *dWg, *dWu, *dh_f, *dh_u;
    float *dg, *du;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dWg, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dWu, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dg, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&du, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dh_f, N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dh_u, N * sizeof(half)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWg, hWg.data(), (size_t)K * N * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dWu, hWu.data(), (size_t)K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(threads), grid((N + threads * VEC - 1) / (threads * VEC), (K + BLOCK_K - 1) / BLOCK_K);
    dim3 eblock(256), egrid((N + 255) / 256);

    auto unfused = [&]() {
        CUDA_CHECK(cudaMemset(dg, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(du, 0, N * sizeof(float)));
        sparse_gemv_kernel<BLOCK_K><<<grid, block>>>(dg, dx, dWg, t, N, K);   // gate
        sparse_gemv_kernel<BLOCK_K><<<grid, block>>>(du, dx, dWu, t, N, K);   // up
        silu_mul_kernel<<<egrid, eblock>>>(dh_u, dg, du, N);
    };
    auto fused = [&]() {
        CUDA_CHECK(cudaMemset(dg, 0, N * sizeof(float)));
        CUDA_CHECK(cudaMemset(du, 0, N * sizeof(float)));
        fused_gate_up_kernel<BLOCK_K><<<grid, block>>>(dg, du, dx, dWg, dWu, t, N, K);
        silu_mul_kernel<<<egrid, eblock>>>(dh_f, dg, du, N);
    };

    unfused(); fused();
    CUDA_CHECK(cudaDeviceSynchronize());

    // Correctness: fused vs unfused, and both vs fp32 CPU reference (first 64).
    std::vector<half> hf(N), hu(N);
    CUDA_CHECK(cudaMemcpy(hf.data(), dh_f, N * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hu.data(), dh_u, N * sizeof(half), cudaMemcpyDeviceToHost));
    double maxrel = 0.0, fu_diff = 0.0;
    for (int n = 0; n < std::min(N, 64); n++) {
        double g = 0, up = 0;
        for (int k = 0; k < K; k++) {
            float xk = __half2float(hx[k]);
            if (fabsf(xk) > t) { g += (double)xk * __half2float(hWg[(size_t)k * N + n]);
                                 up += (double)xk * __half2float(hWu[(size_t)k * N + n]); }
        }
        double ref = (g / (1.0 + exp(-g))) * up;
        maxrel = std::max(maxrel, fabs(ref - __half2float(hf[n])) / (fabs(ref) + 1e-6));
        fu_diff = std::max(fu_diff, fabs((double)__half2float(hf[n]) - __half2float(hu[n])));
    }

    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b));

    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) unfused();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    double t_unf = ev_ms(a, b) / iters;

    CUDA_CHECK(cudaEventRecord(a));
    for (int i = 0; i < iters; i++) fused();
    CUDA_CHECK(cudaEventRecord(b)); CUDA_CHECK(cudaEventSynchronize(b));
    double t_fus = ev_ms(a, b) / iters;

    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak = 300.0;  // L4 HBM GB/s
    double wbytes = 2.0 * active * N * sizeof(half);  // both gate + up weight rows
    auto bw = [&](double ms){ return (wbytes + K * 2 + 2.0 * N * 4) / (ms * 1e-3) / 1e9; };

    printf("device        : %s\n", prop.name);
    printf("gate/up N x K  : %d x %d  (x2 weights)\n", N, K);
    printf("sparsity       : %.0f%%  (%d/%d active)\n", 100.0*(1-(double)active/K), active, K);
    printf("unfused (2+1)  : %.4f ms   %.1f GB/s  %.1f%% bw\n", t_unf, bw(t_unf), 100*bw(t_unf)/peak);
    printf("fused   (1+1)  : %.4f ms   %.1f GB/s  %.1f%% bw\n", t_fus, bw(t_fus), 100*bw(t_fus)/peak);
    printf("fusion speedup : %.3fx\n", t_unf / t_fus);
    printf("fused vs ref   : rel %.2e   fused-vs-unfused max abs %.2e\n", maxrel, fu_diff);
    return 0;
}
