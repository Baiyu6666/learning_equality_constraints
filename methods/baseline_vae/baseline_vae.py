from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn


@dataclass
class TrainConfig:
    epochs: int = 500
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 0.0
    beta_final: float = 0.05
    warmup_epochs: int = 440
    train_log_every: int = 50


class VAEProjectorField(nn.Module):
    """Expose VAE projector as residual field f(x)=x-D(E_mu(x)) for shared APIs."""

    def __init__(self, vae_model: nn.Module):
        super().__init__()
        self.vae_model = vae_model

    def project_tensor(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.vae_model.encode(x)
        return self.vae_model.decode(mu)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.project_tensor(x)


def train_variational_autoencoder(
    model: nn.Module,
    x_train: torch.Tensor,
    cfg: TrainConfig,
    device: torch.device,
) -> dict[str, list[float]]:
    model.train()
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    n = int(x_train.shape[0])
    bs = max(1, int(cfg.batch_size))
    hist: dict[str, list[float]] = {
        "loss_total": [],
        "loss_recon": [],
        "loss_kl": [],
        "beta": [],
    }
    for ep in range(int(cfg.epochs)):
        perm = torch.randperm(n, device=device)
        ep_total = 0.0
        ep_recon = 0.0
        ep_kl = 0.0
        n_steps = 0
        beta = float(cfg.beta_final) * min(1.0, float(ep + 1) / float(max(1, int(cfg.warmup_epochs))))
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            xb = x_train[idx]
            opt.zero_grad(set_to_none=True)
            x_hat, mu, logvar = model(xb)
            recon = ((xb - x_hat) ** 2).mean()
            kl = -0.5 * torch.mean(1.0 + logvar - mu.pow(2) - logvar.exp())
            loss = recon + beta * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            ep_total += float(loss.detach().cpu())
            ep_recon += float(recon.detach().cpu())
            ep_kl += float(kl.detach().cpu())
            n_steps += 1

        if n_steps > 0:
            hist["loss_total"].append(ep_total / n_steps)
            hist["loss_recon"].append(ep_recon / n_steps)
            hist["loss_kl"].append(ep_kl / n_steps)
            hist["beta"].append(beta)
        if ((ep + 1) % max(1, int(cfg.train_log_every))) == 0 or ep == 0 or (ep + 1) == int(cfg.epochs):
            print(
                f"[train] method=vae ep={ep+1:4d}/{cfg.epochs} | lr={float(cfg.lr):.2e} "
                f"| loss={hist['loss_total'][-1]:.6f} | recon={hist['loss_recon'][-1]:.6f} "
                f"| kl={hist['loss_kl'][-1]:.6f} | beta={beta:.3f}"
            )
    return hist


def sample_prior_decode(
    model: nn.Module,
    *,
    latent_dim: int,
    n_sample: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    z = rng.normal(size=(int(n_sample), int(latent_dim))).astype(np.float32)
    with torch.no_grad():
        x_dec = model.decode(torch.from_numpy(z).to(device)).detach().cpu().numpy().astype(np.float32)
    return z, x_dec


def encode_mu(model: nn.Module, x: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        mu, _ = model.encode(torch.from_numpy(x.astype(np.float32)).to(device))
    return mu.detach().cpu().numpy().astype(np.float32)
