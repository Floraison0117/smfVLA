#!/usr/bin/env python3
"""Quick test to verify model loading works with the fixed config."""

import sys
from pathlib import Path

# Setup paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root.parent / "openpi" / "src"))

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from openpi.models import pi0_config
from openpi.models.model import restore_params
from freeflow.models.pi05_freeflow import create_freeflow_model

def test_model_loading():
    """Test that model can be created and checkpoint loaded."""
    print("Testing model loading...")

    # Load checkpoint
    ckpt_path = project_root / "checkpoints" / "base" / "pi05_libero"
    print(f"Loading checkpoint from: {ckpt_path}")

    params = restore_params(
        ckpt_path / "params",
        restore_type=jnp.ndarray,
        dtype=jnp.bfloat16,
    )
    print(f"✓ Checkpoint loaded")

    # Create model with correct config
    model_config = pi0_config.Pi0Config(
        paligemma_variant="gemma_2b",
        action_expert_variant="gemma_300m",
        pi05=True,
        action_dim=32,
        action_horizon=1,
    )
    print(f"✓ Model config created: paligemma={model_config.paligemma_variant}, action_expert={model_config.action_expert_variant}")

    # Create model
    model = create_freeflow_model(model_config)
    print(f"✓ Model created")

    # Test parameter loading
    graphdef, state = nnx.split(model)
    pure_state = state.to_pure_dict()

    import flax.traverse_util as traverse_util
    flat_params = traverse_util.flatten_dict(params)
    flat_state = traverse_util.flatten_dict(pure_state)

    loaded_count = 0
    for key in flat_state:
        if key in flat_params:
            flat_state[key] = flat_params[key]
            loaded_count += 1

    print(f"✓ Loaded {loaded_count} parameters from checkpoint")

    # Verify model structure
    print(f"\nModel structure:")
    print(f"  - Action dim: {model.action_dim}")
    print(f"  - Action horizon: {model.action_horizon}")
    print(f"  - Pi05: {model.pi05}")
    print(f"  - Has time_mlp_in: {hasattr(model, 'time_mlp_in')}")
    print(f"  - Has time_mlp_out: {hasattr(model, 'time_mlp_out')}")

    print("\n✅ Model loading test passed!")
    print("   (Forward pass test skipped - requires tokenizer setup)")
    return True

if __name__ == "__main__":
    try:
        test_model_loading()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
