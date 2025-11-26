# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

# Directions to evaluate a new model checkpoint

# Expected directory structure for new model checkpoints

```
models_to_evaluate/
└── pi05_libero_pytorch_base/
    ├── model.safetensors
    └── assets/
        └── arx/
            └── norm_stats.json
```

## Expected Structure

Ensure models follow this structure

- **`models_to_evaluate/`** - Top-level directory containing models for evaluation
  - **`pi05_libero_pytorch_base/`** - Model directory
    - `model.safetensors` - Model weights in safetensors format
    - **`assets/`** - Assets directory containing model files
      - **`arx/`** - Model-specific subdirectory
        - `norm_stats.json` - Normalization statistics for the model (super important to reproduce results and optimize performance; based on data trained on; if reproducing results, copy corresponding norm_stats.json from the checkpoint dir provided)

1. Put the model checkpoint in the `models_to_evaluate/` directory
2. Put the norm_stats.json file in the `assets/arx/` directory
  - norm_stats.json file should be based on the data used to train the model. For all LIBERO models, just use copy from the other dirs e.g `cp models_to_evaluate/pi05_libero_pytorch_base/assets/arx/norm_stats.json models_to_evaluate/pi05_depth_anything_libero_pytorch_base/assets/arx`
3. Edit the `scripts/serve_policy.py` file to use the new model checkpoint
  # Edit the `DEFAULT_CHECKPOINT` dictionary with the new model checkpoint:
  ```python
  DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="models_to_evaluate/pi05_depth_anything_libero_pytorch_base",
    ),
  }
  ```
4. Run the evaluation script:

```bash
# Download the model checkpoint
uvx hf download griffinlabs/pi05_depth_anything_libero_pytorch_base model.safetensors --local-dir models_to_evaluate/pi05_depth_anything_libero_pytorch_base
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

# Old Instructions for running the evaluation script (for reference)

## With Docker (recommended)

```bash
# Run with default settings (uses Xvfb for headless rendering): -> Griffin Labs Engineers: run git command above, ensure model is in correct directory structure above and then this should work, run in root dir with docker
# edit serve_policy.py to use pi05_libero_finetuned or other checkpoints; put checkpoints in models_to_evaluate/{HF_model_ID}/model.safetensors -> only file required
# Ensure Training Config in src/openpi/training/config.py matches the checkpoint you are using
# Ignore the private macro file robosuite warning, ignore LibGL file not found error
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build


# IMPORTANT: only try these as last resort, if the above doesn't work
# If you get rendering errors, try with EGL instead:
MUJOCO_GL=egl SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# Or with OSMesa (software rendering, slower but most compatible):
MUJOCO_GL=osmesa SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
