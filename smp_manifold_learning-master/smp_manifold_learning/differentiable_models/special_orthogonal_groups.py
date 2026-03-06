# File: special_orthogonal_groups.py
# Torch implementation for differentiable SO(n) optimization.

from __future__ import annotations

import numpy as np
import torch


def convert_to_skewsymm(batch_params: np.ndarray) -> np.ndarray:
    batch_params = np.asarray(batch_params, dtype=np.float64)
    n_batch = int(batch_params.shape[0])
    i = 2
    while int(round((i * (i - 1)) / 2)) < int(batch_params.shape[1]):
        i += 1
    if int(round((i * (i - 1)) / 2)) != int(batch_params.shape[1]):
        raise ValueError("Skew-symmetricity dimension is not found")
    n = i
    out = np.zeros((n_batch, n, n), dtype=np.float64)
    ii, jj = np.tril_indices(n=n, k=-1, m=n)
    for k, (r, c) in enumerate(zip(ii, jj)):
        v = batch_params[:, k]
        out[:, r, c] = v
        out[:, c, r] = -v
    return out


def _to_skewsymm_torch(batch_params: torch.Tensor, n: int) -> torch.Tensor:
    n_batch = int(batch_params.shape[0])
    out = torch.zeros((n_batch, n, n), dtype=batch_params.dtype, device=batch_params.device)
    ii, jj = np.tril_indices(n=n, k=-1, m=n)
    for k, (r, c) in enumerate(zip(ii, jj)):
        out[:, r, c] = batch_params[:, k]
        out[:, c, r] = -batch_params[:, k]
    return out


class SpecialOrthogonalGroups(object):
    def __init__(self, n, N_batch=1, rand_seed=38):
        torch.manual_seed(int(rand_seed))
        self.N_batch = int(N_batch)
        self.n = int(n)
        if self.n < 1:
            raise ValueError("n must be >= 1")
        self.dim_params = int(round((self.n * (self.n - 1)) / 2))
        self.dtype = torch.float32
        self.device = torch.device("cpu")
        if self.dim_params > 0:
            self.params = [
                torch.nn.Parameter(
                    torch.randn((self.dim_params,), dtype=self.dtype, device=self.device) * 1.0e-7
                )
                for _ in range(self.N_batch)
            ]
        else:
            self.params = None
        self._last_transforms = np.tile(np.eye(self.n, dtype=np.float64), (self.N_batch, 1, 1))

    def _transforms_torch(self) -> torch.Tensor:
        if self.params is None:
            return torch.ones((self.N_batch, 1, 1), dtype=self.dtype, device=self.device)
        stacked = torch.stack(self.params, dim=0)
        skew = _to_skewsymm_torch(stacked, n=self.n)
        return torch.linalg.matrix_exp(skew)

    def __call__(self):
        tfm = self._transforms_torch().detach().cpu().numpy()
        self._last_transforms = tfm.astype(np.float64, copy=False)
        return tfm

    def loss(self, target_y, predicted_y):
        target_y = np.asarray(target_y, dtype=np.float64)
        predicted_y = np.asarray(predicted_y, dtype=np.float64)
        eye = np.tile(np.eye(int(target_y.shape[2]), dtype=np.float64), (int(target_y.shape[0]), 1, 1))
        return np.mean(np.mean((eye - (np.transpose(predicted_y, axes=(0, 2, 1)) @ target_y)) ** 2, axis=2), axis=1)

    def _loss_torch(self, target_y: torch.Tensor, predicted_y: torch.Tensor) -> torch.Tensor:
        eye = torch.eye(int(target_y.shape[2]), dtype=target_y.dtype, device=target_y.device).unsqueeze(0).repeat(
            int(target_y.shape[0]), 1, 1
        )
        return torch.mean(torch.mean((eye - (predicted_y.transpose(1, 2) @ target_y)) ** 2, dim=2), dim=1)

    def train(self, inputs, target_outputs, learning_rate=0.001, is_using_separate_opt_per_data_point=True):
        inputs_t = torch.as_tensor(inputs, dtype=self.dtype, device=self.device)
        target_t = torch.as_tensor(target_outputs, dtype=self.dtype, device=self.device)

        if self.params is None:
            transforms = self._transforms_torch()
            outputs = inputs_t @ transforms
            losses = self._loss_torch(target_t, outputs)
            return (
                [float(v) for v in losses.detach().cpu().numpy().tolist()],
                float(losses.mean().detach().cpu().item()),
                transforms.detach().cpu().numpy(),
                outputs.detach().cpu().numpy(),
            )

        if is_using_separate_opt_per_data_point:
            opts = [torch.optim.RMSprop([self.params[i]], lr=float(learning_rate)) for i in range(self.N_batch)]
        else:
            opts = [torch.optim.RMSprop(self.params, lr=float(learning_rate))]

        transforms = self._transforms_torch()
        outputs = inputs_t @ transforms
        losses = self._loss_torch(target_t, outputs)
        mean_loss = losses.mean()

        if is_using_separate_opt_per_data_point:
            for opt in opts:
                opt.zero_grad(set_to_none=True)
            for i in range(self.N_batch):
                retain = i < (self.N_batch - 1)
                losses[i].backward(retain_graph=retain)
                opts[i].step()
                opts[i].zero_grad(set_to_none=True)
        else:
            opt = opts[0]
            opt.zero_grad(set_to_none=True)
            mean_loss.backward()
            opt.step()

        transforms_np = transforms.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()
        losses_np = losses.detach().cpu().numpy()
        self._last_transforms = transforms_np.astype(np.float64, copy=False)
        return (
            [float(v) for v in losses_np.tolist()],
            float(np.mean(losses_np)),
            transforms_np,
            outputs_np,
        )
