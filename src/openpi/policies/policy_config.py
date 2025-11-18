import logging
import os
import pathlib
from typing import Any

import jax.numpy as jnp

import openpi.models.model as _model
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config
import openpi.transforms as transforms


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    
    # Use pytorch_weight_path from config if set, otherwise use checkpoint_dir for weights
    weight_dir = None
    if train_config.pytorch_weight_path is not None:
        weight_path_str = train_config.pytorch_weight_path
        
        # Convert absolute paths that reference project root to container paths
        # Docker compose mounts $PWD to /app, and WORKDIR is /app
        if os.getenv("IS_DOCKER") == "true" and os.path.isabs(weight_path_str):
            if "/openpi/" in weight_path_str:
                # Convert /home/*/openpi/... to /app/...
                parts = weight_path_str.split("/openpi/", 1)
                if len(parts) == 2:
                    weight_path_str = f"/app/{parts[1]}"
                    logging.info(f"Converting pytorch_weight_path to container path: {weight_path_str}")
        
        # Resolve path (relative paths work automatically since WORKDIR=/app in container)
        weight_dir = pathlib.Path(weight_path_str).resolve()
        if not weight_dir.exists():
            logging.warning(f"pytorch_weight_path {weight_dir} does not exist, falling back to checkpoint_dir")
            weight_dir = None
    
    # Try to download/validate checkpoint_dir (needed for norm stats and other assets)
    checkpoint_dir_str = str(checkpoint_dir)
    
    # Resolve relative paths before trying to download
    # (maybe_download handles GCS paths, but relative paths need resolution)
    if not checkpoint_dir_str.startswith(("gs://", "http://", "https://")):
        checkpoint_path = pathlib.Path(checkpoint_dir_str)
        if not checkpoint_path.is_absolute():
            checkpoint_dir_str = str(checkpoint_path.resolve())
    
    try:
        checkpoint_dir = download.maybe_download(checkpoint_dir_str)
    except FileNotFoundError:
        # If checkpoint_dir doesn't exist but we have pytorch_weight_path, try to continue
        # but warn that norm stats might not be available
        if weight_dir is not None:
            logging.warning(f"checkpoint_dir {checkpoint_dir_str} not found, but pytorch_weight_path is set. "
                          f"Norm stats will be loaded from config assets if available.")
            checkpoint_dir = pathlib.Path(checkpoint_dir_str)  # Use original path as Path object
        else:
            raise
    
    # Determine weight path: use pytorch_weight_path if set, otherwise check checkpoint_dir
    if weight_dir is not None:
        weight_path = os.path.join(weight_dir, "model.safetensors")
        is_pytorch = os.path.exists(weight_path)
    else:
        weight_path = os.path.join(checkpoint_dir, "model.safetensors")
        is_pytorch = os.path.exists(weight_path)

    logging.info("Loading model...")
    if is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        # Try to load from checkpoint_dir first, fall back to config assets if checkpoint_dir doesn't exist
        try:
            if isinstance(checkpoint_dir, pathlib.Path) and checkpoint_dir.exists():
                norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)
            else:
                raise FileNotFoundError("checkpoint_dir does not exist")
        except (FileNotFoundError, AttributeError):
            # Fall back to config assets if checkpoint_dir doesn't exist or doesn't have assets
            logging.info(f"Norm stats not found in checkpoint_dir, loading from config assets: {train_config.assets_dirs}")
            norm_stats = data_config.norm_stats

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )
