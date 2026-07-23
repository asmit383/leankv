#!/usr/bin/env bash
# One-shot setup for the int4 end-to-end tok/s test on a fresh L4.
# Requires HF_TOKEN exported (Mistral-7B is gated). ~10-15 min (torch + 14GB model).
set -e

echo "=== 1/3  CUDA compiler (for the torch extension) ==="
apt-get update -qq
apt-get install -y -qq cuda-nvcc-12-4 cuda-cudart-dev-12-4
export PATH=/usr/local/cuda-12.4/bin:$PATH

echo "=== 2/3  Python deps ==="
pip3 install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install -q "transformers==4.44.2" accelerate sentencepiece protobuf ninja

echo "=== 3/3  Mistral-7B-v0.3 (needs HF_TOKEN) ==="
[ -z "$HF_TOKEN" ] && { echo "!! export HF_TOKEN first"; exit 1; }
huggingface-cli download mistralai/Mistral-7B-v0.3 --exclude "*.pt" "consolidated.safetensors"

echo ""
echo "SETUP DONE. Now run:"
echo "  export PATH=/usr/local/cuda-12.4/bin:\$PATH"
echo "  python3 bench_int4_forward.py --model mistralai/Mistral-7B-v0.3 --ntok 128"
