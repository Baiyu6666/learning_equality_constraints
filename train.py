#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass

import numpy as np
import torch
import torch.optim as optim

from models import AutoEncoder, VAE
from utils import batch_iter


@dataclass
class TrainConfig:
    epochs: int = 400
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    # VAE only
    beta_final: float = 1.0
    warmup_epochs: int = 200


def train_ae(model: AutoEncoder, x_train: torch.Tensor, cfg: TrainConfig, device: torch.device) -> None:
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        losses = []
        for xb in batch_iter(x_train, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            x_hat = model(xb)
            loss = ((xb - x_hat) ** 2).mean()
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if (ep + 1) % max(1, cfg.epochs // 10) == 0:
            print(f"[AE] epoch {ep + 1:4d}/{cfg.epochs}  loss={np.mean(losses):.6f}")


def train_vae(model: VAE, x_train: torch.Tensor, cfg: TrainConfig, device: torch.device) -> None:
    model.train()
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    for ep in range(cfg.epochs):
        losses = []
        recon_losses = []
        kl_losses = []
        beta = cfg.beta_final * min(1.0, (ep + 1) / max(1, cfg.warmup_epochs))
        for xb in batch_iter(x_train, cfg.batch_size, shuffle=True):
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(xb)
            recon = ((xb - x_hat) ** 2).mean()
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward()
            opt.step()

            losses.append(float(loss.detach().cpu()))
            recon_losses.append(float(recon.detach().cpu()))
            kl_losses.append(float(kl.detach().cpu()))
        if (ep + 1) % max(1, cfg.epochs // 10) == 0:
            print(
                f"[VAE] epoch {ep + 1:4d}/{cfg.epochs}  loss={np.mean(losses):.6f} "
                f"recon={np.mean(recon_losses):.6f}  kl={np.mean(kl_losses):.6f}  beta={beta:.3f}"
            )
