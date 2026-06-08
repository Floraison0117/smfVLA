"""
SnapFlow Implementation Workflow

Orchestrates multiple agents working in parallel to implement:
1. Training infrastructure (jax_trainer.py, run_train.py, train.sh)
2. Evaluation infrastructure (eval_direct.py, eval_utils.py)
3. Data loader setup
4. Integration testing
"""

export const meta = {
  name: 'snapflow-implementation',
  description: 'Implement SnapFlow training and evaluation infrastructure',
  phases: [
    { title: 'Training Infrastructure', detail: 'Implement jax_trainer.py and training scripts' },
    { title: 'Evaluation Infrastructure', detail: 'Implement eval scripts' },
    { title: 'Data Loader Setup', detail: 'Setup data loading pipeline' },
    { title: 'Integration', detail: 'Verify all components work together' }
  ]
}

async function run() {
  phase('Training Infrastructure')

  log('Starting parallel implementation of SnapFlow components...')

  const trainerCode = await agent(`
Implement src/snapflow/training/jax_trainer.py for SnapFlow training.

Requirements:
1. Adapt from smfVLA/src/smf_vla/training/jax_trainer.py
2. Use SnapFlow loss from snapflow.training.snapflow_loss
3. Support the config structure in configs/train/snapflow_libero.yaml
4. Key features:
   - JAX JIT-compiled training loop
   - Selective parameter updates (trainable params only)
   - Optax AdamW with linear warmup + cosine decay
   - Checkpoint save/load with optimizer state
   - WandB logging for SnapFlow metrics (loss_total, loss_fm, loss_shortcut)
   - Gradient checkpointing for memory efficiency

Use the Pi05SnapFlow model and compute_snapflow_loss function.

The trainer should:
1. Load YAML config
2. Create data loader (reusing from smfVLA)
3. Initialize model from checkpoint
4. Run training loop for 30k steps
5. Save checkpoints every 5k steps
6. Log metrics every 100 steps

Write the complete implementation to /root/autodl-tmp/snapflow/src/snapflow/training/jax_trainer.py
`, { label: 'jax-trainer', phase: 'Training Infrastructure' })

  const runTrainCode = await agent(`
Implement scripts/run_train.py as the entry point for SnapFlow training.

Requirements:
1. Parse command line arguments (config path, resume from checkpoint)
2. Set up PYTHONPATH for imports
3. Load YAML config
4. Initialize WandB run
5. Create trainer and start training
6. Handle checkpoint resumption

The script should be callable like:
  python scripts/run_train.py configs/train/snapflow_libero.yaml
  python scripts/run_train.py configs/train/snapflow_libero.yaml --resume checkpoints/finetuned/snapflow/step_10000

Write the complete implementation to /root/autodl-tmp/snapflow/scripts/run_train.py
`, { label: 'run-train', phase: 'Training Infrastructure' })

  const trainShellCode = await agent(`
Implement scripts/train.sh as the shell wrapper for SnapFlow training.

Requirements:
1. Activate conda environment (openpi_server)
2. Set PYTHONPATH to include src/, openpi/src/, openpi/packages/openpi-client/src/
3. Call run_train.py with passed arguments
4. Handle the project root path correctly

Example:
  bash scripts/train.sh configs/train/snapflow_libero.yaml

Write the implementation to /root/autodl-tmp/snapflow/scripts/train.sh
Make it executable.
`, { label: 'train-shell', phase: 'Training Infrastructure' })

  phase('Evaluation Infrastructure')

  const evalUtilsCode = await agent(`
Implement src/snapflow/eval/eval_utils.py with shared evaluation utilities.

Requirements:
1. Adapt from smfVLA/scripts/eval_utils.py
2. Functions needed:
   - load_policy(nfe, checkpoint_dir, use_smf=False, use_snapflow=True)
   - run_single_task_episode() - core evaluation loop
   - build_result_json(), save_result_json()
   - MAX_STEPS_MAP for LIBERO suites
   - setup_paths() for PYTHONPATH

Key difference from smfVLA:
- Support SnapFlow model (Pi05SnapFlow)
- 1-NFE inference with s=0 (not s=t)

Write the complete implementation to /root/autodl-tmp/snapflow/src/snapflow/eval/eval_utils.py
`, { label: 'eval-utils', phase: 'Evaluation Infrastructure' })

  const evalDirectCode = await agent(`
Implement scripts/eval_direct.py for SnapFlow LIBERO evaluation.

Requirements:
1. Adapt from smfVLA/scripts/eval_direct.py
2. Support presets: quick (spatial, 5 ep), full (all suites, 50 ep)
3. Key arguments:
   --preset {quick,full}
   --nfe {1,10}
   --task-suite {libero_spatial,libero_object,libero_goal,libero_10}
   --num-episodes
   --checkpoint

4. Use SnapFlow model for 1-NFE inference
5. Report success rates per suite

Example:
  python scripts/eval_direct.py --preset quick --nfe 1

Write the complete implementation to /root/autodl-tmp/snapflow/scripts/eval_direct.py
`, { label: 'eval-direct', phase: 'Evaluation Infrastructure' })

  phase('Data Loader Setup')

  const dataLoaderCode = await agent(`
Implement src/snapflow/training/data_loader.py for SnapFlow training.

Requirements:
1. Can either symlink to smfVLA or create a custom version
2. Must support LeRobot v2.0 format (Parquet files)
3. Load from data/libero/ (40 tasks, 1693 episodes)
4. Image preprocessing:
   - Rotate 180°
   - Resize 256x256 to 224x224
5. Return batches with: observation, actions, prompt

Check if smfVLA/src/smf_vla/training/data_loader.py can be reused.
If yes, create a symlink. If no, implement a compatible version.

Write/symlink to /root/autodl-tmp/snapflow/src/snapflow/training/data_loader.py
`, { label: 'data-loader', phase: 'Data Loader Setup' })

  // Wait for all implementations
  phase('Integration')

  const results = await parallel([
    () => trainerCode,
    () => runTrainCode,
    () => trainShellCode,
    () => evalUtilsCode,
    () => evalDirectCode,
    () => dataLoaderCode,
  ])

  const implementations = results.filter(x => x !== null)

  log('Completed ' + implementations.length + '/6 implementation tasks')

  // Integration tests
  const importTest = await agent(`
Verify that all SnapFlow modules can be imported correctly.

Test:
1. cd /root/autodl-tmp/snapflow
2. Activate openpi_server conda environment
3. Try importing:
   - from snapflow.models import Pi05SnapFlow, TargetTimeMLP
   - from snapflow.training import compute_snapflow_loss
   - from snapflow.training.jax_trainer import SnapFlowTrainer
   - from snapflow.eval import eval_utils

4. Report any import errors or missing dependencies

Run this test and report the results.
`, { label: 'import-test', phase: 'Integration' })

  const configTest = await agent(`
Verify that the training config can be loaded and parsed.

Test:
1. Load /root/autodl-tmp/snapflow/configs/train/snapflow_libero.yaml
2. Verify all required fields are present
3. Check that values match expected ranges:
   - alpha: 0.5
   - lambda_consistency: 0.1
   - learning_rate: 2.5e-5
   - training_steps: 30000
   - batch_size: 4

Report any missing or incorrect values.
`, { label: 'config-test', phase: 'Integration' })

  const modelTest = await agent(`
Verify that Pi05SnapFlow model can be initialized.

Test:
1. Load base checkpoint from checkpoints/base/pi05_libero
2. Initialize Pi05SnapFlow model
3. Verify that target_time_mlp is added
4. Check that target_time_mlp outputs zeros (zero initialization)
5. Verify model forward pass works with dummy inputs

Report any initialization errors.
`, { label: 'model-test', phase: 'Integration' })

  // Summary
  log('Implementation Summary:')
  log('Training: ' + (trainerCode && runTrainCode && trainShellCode ? 'OK' : 'FAIL'))
  log('Evaluation: ' + (evalUtilsCode && evalDirectCode ? 'OK' : 'FAIL'))
  log('Data Loader: ' + (dataLoaderCode ? 'OK' : 'FAIL'))
  log('Tests: ' + (importTest && configTest && modelTest ? 'OK' : 'FAIL'))

  return {
    status: 'complete',
    implementations,
    tests: { imports: importTest, config: configTest, model: modelTest }
  }
}

run()
