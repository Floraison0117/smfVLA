#!/bin/bash
# CALVIN 评估环境激活脚本
# 环境: /root/autodl-tmp/conda_envs/calvin_eval (Python 3.9)

# 设置 conda 环境
export CONDA_ENVS_PATH=/root/autodl-tmp/conda_envs
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/calvin_eval

# 设置 PYTHONPATH
export PYTHONPATH="/root/autodl-tmp/datasets/calvin/calvin:/root/autodl-tmp/datasets/calvin/calvin/calvin_env:/root/autodl-tmp/datasets/calvin/calvin/calvin_models:/root/autodl-tmp/eval/scripts:/root/autodl-tmp/smfVLA/src:/root/autodl-tmp/snapflow/src:/root/autodl-tmp/openpi/src:/root/autodl-tmp/openpi/packages/openpi-client/src:$PYTHONPATH"

# 设置评估路径
export EVAL_ROOT=/root/autodl-tmp/eval
export PROJECT_ROOT=/root/autodl-tmp

echo "========================================"
echo "CALVIN Eval Environment"
echo "========================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "Conda env: calvin_eval (Python 3.9)"
echo "PYTHONPATH set for:"
echo "  - CALVIN env"
echo "  - SMF/SnapFlow/openpi"
echo ""
echo "Usage:"
echo "  source activate_calvin_env.sh"
echo "  python eval_calvin.py --help"
echo "========================================"
