#!/bin/bash
#SBATCH --job-name=antiobf-dinosaurs
#SBATCH --account=nn12233k
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail

# Supply a fresh exported code snapshot and absolute JSON config for each job.
code=$(realpath "$1")
config=$(realpath "$2")
mode=${3:-train}
storage=${ANTIOBF_OLIVIA_ROOT:-/cluster/projects/nn12233k/$USER/antiobfbench}
scratch=/cluster/work/projects/nn12233k/$USER/antiobfbench
image=$storage/containers/sglang-v0.5.10.post1-arm64.sif
overlay=$storage/containers/extras.img
cd "$code"
export APPTAINER_CACHEDIR=$scratch/cache APPTAINER_TMPDIR=$scratch/tmp
mkdir -p "$scratch/cache" "$scratch/tmp" "$storage/runs"
slurm_visible=${CUDA_VISIBLE_DEVICES:?Slurm must assign a GPU}
client_visible=$slurm_visible
mps_env=(ANTIOBF_MPS=0)
gpu_mode=$(nvidia-smi -i "$slurm_visible" --query-gpu=compute_mode --format=csv,noheader)
if [ "$gpu_mode" = Exclusive_Process ]; then
    # MPS shares only this job's allocated GPU between SGLang and the optimizer.
    gpu_uuid=$(nvidia-smi -i "$slurm_visible" --query-gpu=uuid --format=csv,noheader)
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/antiobf-mps-$UID-$SLURM_JOB_ID
    export CUDA_MPS_LOG_DIRECTORY=$storage/slurm/mps-$SLURM_JOB_ID
    mkdir -m 700 "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
    cleanup_mps() { printf 'quit\n' | nvidia-cuda-mps-control || true; rm -rf "$CUDA_MPS_PIPE_DIRECTORY"; }
    trap cleanup_mps EXIT
    CUDA_VISIBLE_DEVICES="$gpu_uuid" nvidia-cuda-mps-control -d
    # MPS exposes its single UUID-selected GPU as ordinal zero to clients.
    client_visible=0
    mps_env=(ANTIOBF_MPS=1 CUDA_MPS_PIPE_DIRECTORY="$CUDA_MPS_PIPE_DIRECTORY")
fi

apptainer exec --nv --cleanenv --bind /cluster --overlay "$overlay:ro" "$image" env \
    CUDA_VISIBLE_DEVICES="$client_visible" "${mps_env[@]}" \
    ANTIOBF_SLURM_CUDA_VISIBLE_DEVICES="$slurm_visible" \
    SLURM_JOB_ID="$SLURM_JOB_ID" SLURM_JOB_GPUS="${SLURM_JOB_GPUS:-}" \
    ANTIOBF_MACHINE=olivia ANTIOBF_RUNS_DIR="$storage/runs" \
    ANTIOBF_WANDB_KEY_FILE="/cluster/home/$USER/.antiobfbench_wandb" \
    ANTIOBF_CONTAINER_IMAGE="$image" ANTIOBF_CONTAINER_SHA256="$(cut -d' ' -f1 "$image.sha256")" \
    PYTHONPATH="/opt/antiobfbench:$code/src:$code/rl_trainer/src" PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 TOKENIZERS_PARALLELISM=false \
    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$scratch/cache/huggingface" \
    TORCH_HOME="$scratch/cache/torch" TORCHINDUCTOR_CACHE_DIR="$scratch/cache/inductor" \
    TRITON_CACHE_DIR="$scratch/cache/triton" TORCHINDUCTOR_COMPILE_THREADS=2 \
    FLASHINFER_WORKSPACE_BASE="$scratch/cache/flashinfer" \
    TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda" \
    SGLANG_DG_CACHE_DIR="$scratch/cache/deep_gemm" \
    XDG_CACHE_HOME="$scratch/cache" TMPDIR="$scratch/tmp" \
    http_proxy=http://10.63.2.48:3128 https_proxy=http://10.63.2.48:3128 \
    HTTP_PROXY=http://10.63.2.48:3128 HTTPS_PROXY=http://10.63.2.48:3128 \
    no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1 \
    bash -s -- "$config" "$mode" <<'WORKLOAD'
set -uo pipefail
python3 rl_trainer/scripts/train.py --config "$1"
result=$?
if [ "$result" = 0 ] && [ "$2" = smoke ]; then
    python3 rl_trainer/scripts/test_core.py && python3 rl_trainer/scripts/test_wandb.py
    result=$?
fi
# Let the independent logger flush after training; this never delays an optimizer step.
python3 - "$1" <<'PY'
import json,os,sys,time
from pathlib import Path
name=json.loads(Path(sys.argv[1]).read_text())['run']['name']
status=Path(os.environ['ANTIOBF_RUNS_DIR'])/name/'wandb/status.json'
for _ in range(30):
    if status.exists() and json.loads(status.read_text()).get('state') in ('finished','failed','unavailable'):
        break
    time.sleep(1)
PY
exit "$result"
WORKLOAD
