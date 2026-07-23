#!/usr/bin/env bash
# One-shot MI300X run: compile the HIP sparse GEMV, measure bandwidth ceiling
# (dense) and the sparse path, and check correctness. Minimizes GPU wall-clock.
set -e

echo "=== GPU ==="
rocm-smi --showproductname 2>/dev/null | grep -i "card series\|series" || rocminfo | grep -i "gfx" | head -1

if ! command -v hipcc >/dev/null; then
  echo "hipcc not found — this box needs ROCm. On AMD/DigitalOcean GPU droplets it is preinstalled."
  exit 1
fi

echo -e "\n=== build ==="
hipcc -O3 --offload-arch=gfx942 sparse_gemv_hip.cpp -o sparse_gemv_hip && echo "built"

echo -e "\n=== dense (pure HBM bandwidth ceiling) ==="
./sparse_gemv_hip --sparsity 0.0

echo -e "\n=== 40% activation sparsity (skips weight rows) ==="
./sparse_gemv_hip --sparsity 0.4

echo -e "\nDone. Now DESTROY the droplet in the DigitalOcean panel (not just power off — it bills until destroyed)."
