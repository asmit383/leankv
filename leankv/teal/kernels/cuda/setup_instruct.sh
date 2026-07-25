#!/usr/bin/env bash
# Setup for the instruct + greedy-calibration run. Requires HF_TOKEN exported.
set -e
echo "=== 1/3  CUDA compiler ==="
apt-get update -qq
apt-get install -y -qq cuda-nvcc-12-4 cuda-cudart-dev-12-4
export PATH=/usr/local/cuda-12.4/bin:$PATH
echo "=== 2/3  Python deps ==="
pip3 install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip3 install -q "transformers==4.44.2" accelerate sentencepiece protobuf ninja datasets
echo "=== 3/3  Mistral-7B-Instruct-v0.3 ==="
[ -z "$HF_TOKEN" ] && { echo "!! export HF_TOKEN first"; exit 1; }
huggingface-cli download mistralai/Mistral-7B-Instruct-v0.3 --exclude "*.pt" "consolidated.safetensors"
echo ""
echo "SETUP DONE. Run order:"
echo "  export PATH=/usr/local/cuda-12.4/bin:\$PATH"
echo "  M=mistralai/Mistral-7B-Instruct-v0.3"
echo "  python3 calibrate_simple.py  --model \$M --sparsity 0.40 --out thresholds_uniform.json"
echo "  python3 calibrate_greedy.py  --model \$M --sparsity 0.40 --out thresholds_greedy.json"
echo "  python3 ppl.py --model \$M --thresholds thresholds_uniform.json   # baseline +17%-ish"
echo "  python3 ppl.py --model \$M --thresholds thresholds_greedy.json    # should be lower"
echo "  python3 bench_int4_fused_graph.py --model \$M --thresholds thresholds_greedy.json"
echo "  python3 chat.py --model \$M --thresholds thresholds_greedy.json   # real chat"
