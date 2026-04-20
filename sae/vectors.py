"""
Steering vector save/load utilities.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch


def normalize_vector(vector: torch.Tensor) -> torch.Tensor:
    """Return the unit-normalised version of *vector*."""
    return vector * (1.0 / vector.norm())


def save_vectors(
    steering_vectors: Dict[int, torch.Tensor],
    output_dir: str,
    add_timestamp: bool = True,
) -> str:
    """Save raw and normalised steering vectors to disk.

    Creates ``<output_dir>/raw/layer_N.pt`` and
    ``<output_dir>/normalized/layer_N.pt`` for each layer.

    Args:
        steering_vectors: Dict mapping layer index → steering tensor.
        output_dir: Base directory path (timestamp optionally appended).
        add_timestamp: If True, appends ``_YYYYMMDD_HHMMSS`` to the dir name.

    Returns:
        Path to the directory where vectors were saved.
    """
    if add_timestamp:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = f"{output_dir}_{ts}"
    else:
        base_dir = output_dir

    os.makedirs(os.path.join(base_dir, "raw"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "normalized"), exist_ok=True)

    for layer, vector in steering_vectors.items():
        torch.save(vector, os.path.join(base_dir, "raw", f"layer_{layer}.pt"))
        torch.save(normalize_vector(vector), os.path.join(base_dir, "normalized", f"layer_{layer}.pt"))

    return base_dir


def load_steering_vectors(
    vectors_dir: str,
    normalized: bool = False,
) -> Dict[int, torch.Tensor]:
    """Load steering vectors saved by :func:`save_vectors`.

    Args:
        vectors_dir: Root directory returned by :func:`save_vectors`.
        normalized: If True, load from the ``normalized/`` sub-directory;
            otherwise load from ``raw/``.

    Returns:
        Dict mapping layer index → tensor.
    """
    subdir = "normalized" if normalized else "raw"
    target = Path(vectors_dir) / subdir

    vectors: Dict[int, torch.Tensor] = {}
    for pt_file in sorted(target.glob("layer_*.pt")):
        layer_idx = int(pt_file.stem.split("_")[1])
        vectors[layer_idx] = torch.load(pt_file, map_location="cpu", weights_only=True)

    if not vectors:
        raise FileNotFoundError(
            f"No layer_*.pt files found in {target}. "
            "Make sure the path points to a directory created by save_vectors()."
        )

    return vectors
