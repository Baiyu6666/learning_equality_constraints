#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Iterator

import numpy as np
import torch


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_tensor(x: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def recon_error_l2(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    # per-sample L2 norm
    return torch.norm(x - x_hat, dim=1)


def batch_iter(x: torch.Tensor, batch_size: int, shuffle: bool = True) -> Iterator[torch.Tensor]:
    n = x.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for s in range(0, n, batch_size):
        j = idx[s : s + batch_size]
        yield x[j]
