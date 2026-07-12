#!/bin/bash
# CALVIN ABCD→D benchmark: NFE=10,4,2,1 × num_sequences 条序列。
# 用法: bash run_calvin_benchmark_all_nfe.sh [num_sequences] [ckpt] [dataset]   # 默认 100 / pi05_calvin_corrected / ABCD
#
# 注意：环境用 openpi_server（JAX支持Blackwell sm_120；policy加载链在此env），
#       不是 calvin_eval（缺augmax/jax0.5.3）。CALVIN仿真依赖(pybullet/hydra/numpy-quaternion)已装进openpi_server。

NSEQ="${1:-100}"
CKPT="${2:-/root/autodl-tmp/checkpoints/pi05_calvin_corrected}"
DATASET="${3:-ABCD}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活 openpi_server
source /root/miniconda3/etc/profile.d/conda.sh
conda activate openpi_server

# PYTHONPATH: calvin_env + calvin_models + eval/scripts + openpi + smf/snapflow（让 calvin_env 靠路径导入）
export PYTHONPATH="/root/autodl-tmp/datasets/calvin/calvin:/root/autodl-tmp/datasets/calvin/calvin/calvin_env:/root/autodl-tmp/datasets/calvin/calvin/calvin_models:/root/autodl-tmp/eval/scripts:/root/autodl-tmp/openpi/src:/root/autodl-tmp/openpi/packages/openpi-client/src:/root/autodl-tmp/smfVLA/src:/root/autodl-tmp/snapflow/src:${PYTHONPATH:-}"

# headless 渲染 + JAX 不预分配
export DISPLAY=""
export PYOPENGL_PLATFORM=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYBULLET_EGL="${PYBULLET_EGL:-1}"

cd "$SCRIPT_DIR"
echo "========================================"
echo "CALVIN ABCD→D | NFE=10,4,2,1 | seq=${NSEQ}"
echo "  ckpt   : ${CKPT}"
echo "  dataset: ${DATASET}"
echo "  python : $(python -c 'import sys;print(sys.executable)')"
echo "========================================"

FAILED=()
for NFE in 10 4 2 1; do
  echo ""
  echo "======== NFE=${NFE} (${NSEQ} sequences) ========"
  if python eval_calvin_benchmark.py \
      --dataset "${DATASET}" --checkpoint "${CKPT}" \
      --nfe "${NFE}" --num-sequences "${NSEQ}" --replan-steps 5; then
    echo "======== NFE=${NFE} 完成 ✓ ========"
  else
    echo "======== NFE=${NFE} 失败 ✗ ========"
    FAILED+=("${NFE}")
  fi
done

echo ""
echo "========================================"
echo "全部 NFE 完成。结果: /root/autodl-tmp/eval/results/calvin/"
ls -t /root/autodl-tmp/eval/results/calvin/*calvin_${DATASET}_*.json 2>/dev/null | head -8
if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "⚠ 失败的 NFE: ${FAILED[*]}"
  exit 1
fi
echo "========================================"
