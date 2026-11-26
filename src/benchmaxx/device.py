"""Choosing where to compute."""

import torch


def default_device() -> torch.device:
    """CUDA if available, then Apple MPS, then CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
