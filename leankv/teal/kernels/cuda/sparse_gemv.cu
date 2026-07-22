// sparse_gemv.cu — raw CUDA batch-1 sparse GEMV for TEAL activation sparsity.
//
// Computes y[n] = sum_k ( |x[k]| > threshold ? x[k] * W[k,n] : 0 )
//
// This is the memory-bound batch-1 decode regime: one activation vector x
// multiplied against a full weight matrix W. The kernel's whole job is to move
// W from HBM as fast as possible while skipping the rows whose activation is
// zeroed by TEAL sparsity — every skipped row is weight traffic we never pay.
//
// Weight layout matches the Triton kernel: column-major, i.e. the flat buffer
// is [K, N] row-major so W[k*N + n] == weight[n, k]. Consecutive output columns
// n are contiguous in memory, so a warp's loads for a fixed k are coalesced.
//
// Design:
//   - float4 (128-bit) vectorized loads: 8 half weights per thread per k.
//   - Split-K over the grid.y dimension so we launch enough blocks to fill the
//     SMs and keep enough memory requests in flight to hide HBM latency.
//   - The sparsity check is uniform across a block (all threads read the same
//     x[k] from shared memory), so a skipped k is skipped by the whole block
//     with zero warp divergence.
//   - Partial sums from each K-split are merged into y via atomicAdd (fp32).

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define VEC 8            // half weights per thread per k  (float4 = 128-bit load)
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t _e = (call);                                             \
        if (_e != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                      \
                    cudaGetErrorString(_e), __FILE__, __LINE__);             \
            exit(1);                                                         \
        }                                                                    \
    } while (0)

// ── Kernel ──────────────────────────────────────────────────────────────
// y : [N]    fp32, must be zeroed before launch (split-K accumulates here)
// x : [K]    fp16
// W : [K,N]  fp16 row-major (== column-major weight)
// grid  = ( N / (blockDim.x * VEC),  ceil(K / BLOCK_K) )
// block = blockDim.x threads, each owns VEC contiguous output columns.
template <int BLOCK_K>
__global__ void sparse_gemv_kernel(float *__restrict__ y,
                                   const half *__restrict__ x,
                                   const half *__restrict__ W,
                                   float threshold, int N, int K) {
    __shared__ half xs[BLOCK_K];

    const int k0 = blockIdx.y * BLOCK_K;
    const int klen = min(BLOCK_K, K - k0);

    // Cooperative load of this block's x-slice into shared memory (read once,
    // reused by every thread across all its output columns).
    for (int i = threadIdx.x; i < klen; i += blockDim.x)
        xs[i] = x[k0 + i];
    __syncthreads();

    const int n_base = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (n_base >= N) return;

    float acc[VEC];
#pragma unroll
    for (int j = 0; j < VEC; j++) acc[j] = 0.f;

    for (int k = 0; k < klen; k++) {
        const float xk = __half2float(xs[k]);
        if (fabsf(xk) <= threshold) continue;   // skip the whole W row — bandwidth saved
        const float4 wv =
            *reinterpret_cast<const float4 *>(&W[(size_t)(k0 + k) * N + n_base]);
        const half *wh = reinterpret_cast<const half *>(&wv);
#pragma unroll
        for (int j = 0; j < VEC; j++)
            acc[j] += xk * __half2float(wh[j]);
    }

#pragma unroll
    for (int j = 0; j < VEC; j++)
        atomicAdd(&y[n_base + j], acc[j]);
}

// ── Host benchmark ──────────────────────────────────────────────────────

static double now_ms(cudaEvent_t a, cudaEvent_t b) {
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    return (double)ms;
}

int main(int argc, char **argv) {
    // Defaults sized to Mistral-7B MLP up_proj: K=4096 -> N=14336.
    int N = 14336, K = 4096;
    float sparsity = 0.40f;       // fraction of x entries zeroed (below threshold)
    int threads = 128;            // -> BLOCK_N = 128*8 = 1024 columns per block
    const int BLOCK_K = 512;
    int iters = 200;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--N")) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--K")) K = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--sparsity")) sparsity = atof(argv[++i]);
        else if (!strcmp(argv[i], "--threads")) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--iters")) iters = atoi(argv[++i]);
    }
    if (N % VEC != 0) { fprintf(stderr, "N must be divisible by %d\n", VEC); return 1; }

    const float threshold = 0.5f;

    // Host data. x: a controlled fraction below threshold (sparse), rest above.
    std::mt19937 rng(0);
    std::uniform_real_distribution<float> u(0.f, 1.f);
    std::vector<half> hx(K), hW((size_t)K * N);
    int active = 0;
    for (int k = 0; k < K; k++) {
        float v = (u(rng) < sparsity) ? 0.0f          // zeroed activation
                                      : (1.0f + u(rng)); // magnitude > threshold
        hx[k] = __float2half(v);
        if (fabsf(v) > threshold) active++;
    }
    for (size_t i = 0; i < (size_t)K * N; i++)
        hW[i] = __float2half(u(rng) * 0.02f - 0.01f);

    half *dx, *dW; float *dy;
    CUDA_CHECK(cudaMalloc(&dx, K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dW, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&dy, N * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), K * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dW, hW.data(), (size_t)K * N * sizeof(half), cudaMemcpyHostToDevice));

    dim3 block(threads);
    dim3 grid(N / (threads * VEC), (K + BLOCK_K - 1) / BLOCK_K);
    if (N % (threads * VEC) != 0) grid.x += 1;

    auto launch = [&]() {
        CUDA_CHECK(cudaMemset(dy, 0, N * sizeof(float)));
        sparse_gemv_kernel<BLOCK_K><<<grid, block>>>(dy, dx, dW, threshold, N, K);
    };

    // Warmup + correctness check against CPU reference.
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    std::vector<float> hy(N);
    CUDA_CHECK(cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    double max_err = 0.0;
    for (int n = 0; n < std::min(N, 64); n++) {
        double ref = 0.0;
        for (int k = 0; k < K; k++) {
            float xk = __half2float(hx[k]);
            if (fabsf(xk) > threshold)
                ref += (double)xk * __half2float(hW[(size_t)k * N + n]);
        }
        max_err = std::max(max_err, fabs(ref - hy[n]) / (fabs(ref) + 1e-6));
    }

    // Timed loop.
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));
    double ms = now_ms(t0, t1) / iters;

    // Bandwidth: bytes actually pulled from HBM. Only the active weight rows are
    // loaded (that's the point of the sparsity skip), plus x and the y accumulate.
    double w_bytes = (double)active * N * sizeof(half);
    double bytes = w_bytes + K * sizeof(half) + N * sizeof(float);
    double gbps = bytes / (ms * 1e-3) / 1e9;

    // Peak HBM bandwidth of the device (measured from the memory clock/bus width).
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    double peak_gbps = 2.0 * prop.memoryClockRate * 1e3 * (prop.memoryBusWidth / 8) / 1e9;

    printf("device            : %s\n", prop.name);
    printf("N x K             : %d x %d\n", N, K);
    printf("sparsity          : %.1f%%  (%d/%d rows active)\n",
           100.0 * (1.0 - (double)active / K), active, K);
    printf("grid              : (%d, %d)  block %d\n", grid.x, grid.y, threads);
    printf("time / call       : %.4f ms\n", ms);
    printf("weight traffic    : %.2f MB\n", w_bytes / 1e6);
    printf("effective bw      : %.1f GB/s\n", gbps);
    printf("peak HBM bw       : %.1f GB/s\n", peak_gbps);
    printf("bw utilization    : %.1f%%\n", 100.0 * gbps / peak_gbps);
    printf("max rel err (n<64): %.2e\n", max_err);

    cudaFree(dx); cudaFree(dW); cudaFree(dy);
    return 0;
}
