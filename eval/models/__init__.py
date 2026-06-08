"""
统一的模型加载接口，支持SMF和SnapFlow评估。

This module provides a unified interface for loading different model
checkpoints for evaluation on LIBERO benchmarks.
"""

from .smf_adapter import load_smf_model, load_smf_config
from .snapflow_adapter import load_snapflow_model, load_snapflow_config

__all__ = [
    'load_smf_model',
    'load_smf_config',
    'load_snapflow_model',
    'load_snapflow_config',
]
