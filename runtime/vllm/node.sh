#!/usr/bin/env bash
# Runs inside an image built by the sibling Dockerfile, with one local shard.
set -euo pipefail
: "${NODE_RANK:?set NODE_RANK=0..5}"
: "${HEAD_IP:?set HEAD_IP to the head RoCE fabric address}"
export AMOS_EXL3_TP6_PIECES=1
MODEL_DIR="${MODEL_DIR:-/model}"
PROFILE="${PROFILE:-mtp4}"
DENSE="${DENSE:-bf16}"
DRAFT_DENSE="${DRAFT_DENSE:-bf16}"
METHOD=mtp

case "$NODE_RANK" in 0|1|2|3|4|5) ;; *) echo 'invalid TP6 node rank' >&2; exit 2 ;; esac
case "$PROFILE" in
  target-smoke) LENGTH=8192; SEQS=1; BATCH=256; SPEC=0; EAGER=1 ;;
  mtp4-smoke) LENGTH=32768; SEQS=4; BATCH=1024; SPEC=4; EAGER=0 ;;
  mtp4) LENGTH=360000; SEQS=4; BATCH=1024; SPEC=4; EAGER=0 ;;
  ngram4) LENGTH=360000; SEQS=4; BATCH=1024; SPEC=4; EAGER=0; METHOD=ngram ;;
  ngram-gpu4) LENGTH=360000; SEQS=4; BATCH=1024; SPEC=4; EAGER=0; METHOD=ngram_gpu ;;
  *) echo 'PROFILE must be target-smoke, mtp4-smoke, mtp4, ngram4, or ngram-gpu4' >&2; exit 2 ;;
esac
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[1-9][0-9]*$ ]] &&
    (( MAX_NUM_BATCHED_TOKENS >= 128 && MAX_NUM_BATCHED_TOKENS <= 32768 )) || {
      echo 'MAX_NUM_BATCHED_TOKENS must be an integer in 128..32768' >&2; exit 2;
    }
  BATCH="$MAX_NUM_BATCHED_TOKENS"
fi
case "$DENSE" in bf16|mxfp8) ;; *) echo 'DENSE must be bf16 or mxfp8' >&2; exit 2 ;; esac
case "$DRAFT_DENSE" in bf16|mxfp8) ;; *) echo 'DRAFT_DENSE must be bf16 or mxfp8' >&2; exit 2 ;; esac
case "${AMOS_TP6_E3_PREFILL:-0}" in 0|1) ;; *) echo 'AMOS_TP6_E3_PREFILL must be 0 or 1' >&2; exit 2 ;; esac
if [[ "${AMOS_TP6_E3_PREFILL:-0}" == 1 && "${DRY_RUN:-0}" != 1 ]]; then
  python3 -c 'from vllm import amos_grouped_prefill; from vllm.amos_e3 import runtime'
fi
case "${AMOS_TP6_SHARED_384:-0}" in 0|1) ;; *) echo 'AMOS_TP6_SHARED_384 must be 0 or 1' >&2; exit 2 ;; esac
if [[ "${AMOS_TP6_SHARED_384:-0}" == 1 ]]; then
  [[ "$SPEC" == 4 && "$DENSE" == mxfp8 && "$DRAFT_DENSE" == mxfp8 ]] || {
    echo 'Shared384 requires MTP4 and target/draft MXFP8' >&2; exit 2;
  }
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    python3 -c 'from vllm import amos_shared_expert_padding; assert amos_shared_expert_padding.enabled()'
  fi
fi
case "${AMOS_TP6_DRAFT_EH:-0}" in 0|1) ;; *) echo 'AMOS_TP6_DRAFT_EH must be 0 or 1' >&2; exit 2 ;; esac
if [[ "${AMOS_TP6_DRAFT_EH:-0}" == 1 ]]; then
  [[ "$METHOD" == mtp && "$SPEC" == 4 && "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" == 1 ]] || {
    echo 'Draft EH sharding requires MTP4 and TP6 RoCE collectives' >&2; exit 2;
  }
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    python3 -c 'from vllm import amos_draft_eh; assert amos_draft_eh.enabled()'
  fi
fi
case "${AMOS_TP6_DECODE_PROJECTIONS:-0}" in 0|1) ;; *) echo 'AMOS_TP6_DECODE_PROJECTIONS must be 0 or 1' >&2; exit 2 ;; esac
if [[ "${AMOS_TP6_DECODE_PROJECTIONS:-0}" == 1 ]]; then
  [[ "$DENSE" == mxfp8 && "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" == 1 ]] || {
    echo 'Decode projections require target MXFP8 and TP6 RoCE collectives' >&2; exit 2;
  }
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    python3 -c 'from vllm import amos_decode_projections; assert amos_decode_projections.enabled()'
  fi
fi

if [[ "$NODE_RANK" == 0 ]]; then
  if [[ -n "${API_KEY_FILE:-}" ]]; then
    # Keep the credential out of Docker's config and the process command line.
    VLLM_API_KEY="$(cat "$API_KEY_FILE")"
    [[ -n "$VLLM_API_KEY" ]] || { echo 'API key file is empty' >&2; exit 2; }
    export VLLM_API_KEY
  fi
  if [[ "${API_HOST:-127.0.0.1}" != 127.0.0.1 && -z "${VLLM_API_KEY:-}" ]]; then
    echo 'A non-loopback API requires authentication' >&2; exit 2
  fi
fi

if [[ "${DRY_RUN:-0}" != 1 ]]; then
  python3 - "$MODEL_DIR" "$NODE_RANK" <<'PY'
import json, sys
from pathlib import Path
root, rank = Path(sys.argv[1]), int(sys.argv[2])
p = json.loads((root / 'TP6_PLACEMENT.json').read_text())
v = json.loads((root / 'TP6_VERIFIED.json').read_text())
if p['rank'] != rank or v['rank'] != rank:
    raise SystemExit('local shard belongs to a different rank')
if p['source_revision'] != '6d6bd738c0c1635513e0bd0fdf0302049bd820a9' or v['source_revision'] != p['source_revision']:
    raise SystemExit('unexpected checkpoint revision')
for item in v['files']:
    if (root / item['path']).stat().st_size != item['bytes']:
        raise SystemExit('verified shard changed: ' + item['path'])
PY
fi

ARGS=(
  --served-model-name glm-5.3
  --host "${API_HOST:-127.0.0.1}" --port "${PORT:-8953}"
  --tensor-parallel-size 6 --nnodes 6 --node-rank "$NODE_RANK"
  --master-addr "$HEAD_IP" --master-port "${MASTER_PORT:-29623}"
  --distributed-executor-backend mp --decode-context-parallel-size 1
  --max-model-len "$LENGTH" --max-num-seqs "$SEQS" --max-num-batched-tokens "$BATCH"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}"
  --kv-cache-dtype fp8 --attention-backend B12X_MLA_SPARSE --moe-backend b12x
  --load-format safetensors --safetensors-load-strategy eager --quantization exl3
  --enable-chunked-prefill --enable-prefix-caching
  --no-enable-flashinfer-autotune
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser glm45
  --generation-config vllm
  --hf-overrides '{"index_topk_pattern":"FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS"}'
)
[[ "${VLLM_ENABLE_ROCE_ALLREDUCE:-0}" == 1 ]] || ARGS+=(--disable-custom-all-reduce)
[[ "$NODE_RANK" == 0 ]] || ARGS+=(--headless)
[[ "$EAGER" == 0 ]] || ARGS+=(--enforce-eager)
[[ -z "${KV_CACHE_MEMORY_BYTES:-}" ]] || ARGS+=(--kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES")
case "${AMOS_TORCH_PROFILER:-0}" in 0|1) ;; *) echo 'AMOS_TORCH_PROFILER must be 0 or 1' >&2; exit 2 ;; esac
if [[ "${AMOS_TORCH_PROFILER:-0}" == 1 ]]; then
  # Inactive until authenticated /start_profile; bound trace memory on each rank.
  ARGS+=(--profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/.cache/amos-tp6-profiles","torch_profiler_with_stack":false,"torch_profiler_record_shapes":false,"torch_profiler_with_memory":false,"torch_profiler_use_gzip":true,"ignore_frontend":true,"max_iterations":8}')
fi
if [[ "${AMOS_CAPACITY_MIDDLEWARE:-0}" == 1 && "$NODE_RANK" == 0 ]]; then
  ARGS+=(--middleware vllm.amos_capacity.CapacityMiddleware)
fi
if [[ "$DENSE" == mxfp8 || "$DRAFT_DENSE" == mxfp8 ]]; then
  ARGS+=(--linear-backend b12x)
fi
[[ "$DENSE" != mxfp8 ]] || ARGS+=(
  --quantization-config '{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"}}')
if [[ "$SPEC" == 4 ]]; then
  SPEC_CONFIG="$(python3 - "$DRAFT_DENSE" "$METHOD" <<'PY'
import json, os, sys
if sys.argv[2] in ('ngram','ngram_gpu'):
    disabled = ('AMOS_MTP_ADAPT_WINDOW','AMOS_MTP_FIXED_DEPTH','AMOS_MTP_CALIBRATION',
                'AMOS_MTP_COST_AWARE','AMOS_MTP_REQUEST_PHASE','AMOS_MTP_TUNING_CONTROL',
                'AMOS_TP6_DRAFT_EH')
    if any(os.environ.get(k,'0')!='0' for k in disabled):
        raise SystemExit('Copy drafting requires explicit disabled MTP-only settings')
    if os.environ.get('AMOS_TP6_DYNAMIC_GRAPHS')!='1':
        raise SystemExit('Copy trial requires existing q1..5/C1..4 target graph coverage')
    print(json.dumps({'method':sys.argv[2],'num_speculative_tokens':4,
                      'prompt_lookup_min':5,'prompt_lookup_max':5,
                      'rejection_sample_method':'standard'}))
    raise SystemExit(0)
config = {'method': 'mtp', 'num_speculative_tokens': 4, 'moe_backend': 'b12x',
          'attention_backend': 'B12X_MLA_SPARSE', 'draft_sample_method': 'probabilistic',
          'rejection_sample_method': 'standard'}
graphs = int(os.environ.get('AMOS_TP6_DYNAMIC_GRAPHS', '0'))
depth = int(os.environ.get('AMOS_MTP_FIXED_DEPTH', '0'))
window = int(os.environ.get('AMOS_MTP_ADAPT_WINDOW', '0'))
calibration = int(os.environ.get('AMOS_MTP_CALIBRATION', '0'))
cost_aware = int(os.environ.get('AMOS_MTP_COST_AWARE', '0'))
if graphs not in (0,1) or not 0 <= depth <= 4 or not 0 <= window <= 1024 or calibration not in (0,1):
    raise SystemExit('Invalid dynamic MTP settings')
if (depth or window or calibration) and not graphs:
    raise SystemExit('Dynamic MTP requires all-depth target CUDA graphs')
if sum(bool(v) for v in (depth,window,calibration)) > 1:
    raise SystemExit('Choose fixed-depth measurement or adaptive scheduling')
if cost_aware not in (0,1) or (cost_aware and not 4 <= window <= 128):
    raise SystemExit('Cost-aware MTP requires an adaptive window of 4..128')
if depth:
    config['num_speculative_tokens_per_batch_size'] = [[1,4,depth]]
if calibration:
    config['num_speculative_tokens_per_batch_size'] = [[1,4,4]]
if window:
    config['adaptive_speculative_tokens_window'] = window
if sys.argv[1] == 'mxfp8':
    config['quantization_config'] = {'linear': {'weight': 'mxfp8'},
                                     'shared_experts': {'weight': 'mxfp8'}}
print(json.dumps(config))
PY
)"
  [[ "$METHOD" != ngram ]] || ARGS+=(--no-async-scheduling)
  [[ "$METHOD" != ngram_gpu ]] || ARGS+=(--async-scheduling)
  ARGS+=(
    --speculative-config "$SPEC_CONFIG"
    --compilation-config '{"cudagraph_mode":"FULL","cudagraph_capture_sizes":[5,10,15,20]}'
  )
  if [[ "${AMOS_TP6_DYNAMIC_GRAPHS:-0}" == 1 ]]; then
    # A flag on the original image must fail before loading weights.
    if [[ "${DRY_RUN:-0}" != 1 ]]; then
      python3 -c 'from vllm import amos_dynamic_graphs'
    fi
    ARGS+=(--cudagraph-metrics)
  fi
fi
if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'vllm serve %q' "$MODEL_DIR"
  printf ' %q' "${ARGS[@]}"
  printf '\n'
  exit 0
fi
exec vllm serve "$MODEL_DIR" "${ARGS[@]}"
