#!/usr/bin/env bash
#
# Start Qwen3.8-27B with a tested vLLM profile.
#
# Usage:
#   bash scripts/run_vllm_profile.sh text
#   bash scripts/run_vllm_profile.sh code
#   bash scripts/run_vllm_profile.sh ocr
#   bash scripts/run_vllm_profile.sh long-80k-quality
#   bash scripts/run_vllm_profile.sh long-80k-capacity
#   bash scripts/run_vllm_profile.sh long-120k-quality
#   bash scripts/run_vllm_profile.sh long-120k-capacity
#
# Optional overrides, for example:
#   MODEL=/models/Qwen3.8-27B PORT=8001 bash scripts/run_vllm_profile.sh text

set -euo pipefail

PROFILE="${1:-}"
MODEL="${MODEL:-/root/llm-cache/qwen3.8-27b}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-128}"
MAX_PIXELS="${MAX_PIXELS:-1048576}"  # 1 MP/page: measured OCR profile

if [[ -z "$PROFILE" ]]; then
  echo "Usage: $0 {text|code|ocr|long-80k-quality|long-80k-capacity|long-120k-quality|long-120k-capacity}" >&2
  exit 64
fi

# Required on the tested WSL host when FlashInfer compiles FP8 kernels.
if [[ -d /usr/local/cuda ]]; then
  export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
  export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
fi

# The tested vLLM build was stable with the PyTorch sampler.  This does not
# disable FlashInfer attention selected automatically for FP8 KV cache.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"

case "$PROFILE" in
  text)
    MAX_MODEL_LEN=12288
    KV_CACHE_DTYPE=auto
    ;;
  code|ocr)
    MAX_MODEL_LEN=32768
    KV_CACHE_DTYPE=auto
    ;;
  long-80k-quality)
    MAX_MODEL_LEN=81920
    KV_CACHE_DTYPE=auto
    ;;
  long-80k-capacity)
    MAX_MODEL_LEN=81920
    KV_CACHE_DTYPE=fp8_e4m3
    ;;
  long-120k-quality)
    MAX_MODEL_LEN=122880
    KV_CACHE_DTYPE=auto
    ;;
  long-120k-capacity)
    MAX_MODEL_LEN=122880
    KV_CACHE_DTYPE=fp8_e4m3
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    exit 64
    ;;
esac

echo "Starting profile=$PROFILE max_model_len=$MAX_MODEL_LEN kv_cache_dtype=$KV_CACHE_DTYPE mtp_draft_tokens=3"

exec vllm serve "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --served-model-name qwen3.8-27b \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --limit-mm-per-prompt '{"image":16}' \
  --mm-processor-kwargs "{\"max_pixels\":${MAX_PIXELS}}" \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
