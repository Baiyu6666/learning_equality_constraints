#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AE/VAE Manifold Benchmark (on-manifold only training; eval uses GT on/off sampling)

Features:
1) Multiple datasets:
   - 3D: spiral (1D curve in R3), sphere surface (2D manifold in R3),
         paraboloid surface z=x^2+y^2 (2D manifold in R3),
         two-sphere union outer boundary (2D manifold in R3, keeps exterior surface only)
   - 2D: circle boundary (1D manifold in R2), square boundary (1D manifold in R2)

2) Train AE and/or VAE
3) Evaluate as one-class classifier using recon error ||x - D(E(x))||:
   - classify as ON if error <= threshold
   - compute confusion matrix, acc, precision, recall, f1, AUROC (score = -error)
   - GT on/off samples are generated with fixed seed for reproducibility
4) Plot:
   - latent scatter: encoded eval points (AE: z, VAE: mu), colored by true on/off
   - latent sampling -> decode -> scatter in original space
5) Plot:
   - arrows from original points to projected points P(x)=D(E(x))

Notes:
- "ON manifold" means within threshold under the model’s reconstruction/projection mapping.
- This is not guaranteed to match true manifold globally; it’s a benchmark tool.
"""

import argparse

import numpy as np
import torch

from datasets.vae_datasets import build_datasets
from methods.baseline_vae.evaluate import estimate_threshold, eval_with_threshold
from methods.baseline_vae.models import AutoEncoder, VAE
from methods.baseline_vae.planner import build_planner_cases
from methods.baseline_vae.train import TrainConfig, train_ae, train_vae
from methods.baseline_vae.utils import ensure_dir, recon_error_l2, to_tensor
from methods.baseline_vae.viz import plot_planner_grid, visualize_all


# -----------------------------
# Main pipeline
# -----------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="2d_square", choices=list(build_datasets().keys()))
    parser.add_argument("--models", type=str, default="both", choices=["ae", "vae", "both"])
    parser.add_argument("--latent_dim", type=int, default=-1, help="If -1, use dataset default.")
    parser.add_argument("--train_n", type=int, default=50)
    parser.add_argument("--eval_on_n", type=int, default=500)
    parser.add_argument("--eval_off_n", type=int, default=500)
    parser.add_argument("--sample_latent_n", type=int, default=150)
    parser.add_argument("--train_seed", type=int, default=36)
    parser.add_argument("--eval_seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta_final", type=float, default=0.05)
    parser.add_argument("--warmup", type=int, default=440)
    parser.add_argument(
        "--threshold_q",
        type=float,
        default=95.0,
        help="Percentile of ON errors to set threshold.",
    )
    parser.add_argument("--outdir", type=str, default="outputs_manifold_bench")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto: use cuda if available else cpu",
    )
    parser.add_argument("--planner", type=str, default="on", choices=["on", "off"])
    parser.add_argument("--planner_model", type=str, default="auto", choices=["auto", "ae", "vae"])
    parser.add_argument("--planner_steps", type=int, default=64)
    parser.add_argument("--planner_pairs", type=int, default=4)
    args = parser.parse_args()

    ensure_dir(args.outdir)
    ds = build_datasets()[args.dataset]

    latent_dim = ds.latent_dim_default if args.latent_dim < 0 else args.latent_dim

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Check torch install and nvidia-smi.")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"[Device] Using {device}")

    # ---------------- Train data (ON only) ----------------
    rng_train = np.random.default_rng(args.train_seed)
    x_train = ds.train_on_sampler(args.train_n, rng_train)
    x_train_t = to_tensor(x_train, device)

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        beta_final=args.beta_final,
        warmup_epochs=args.warmup,
    )

    # Validation data
    x_val_on = ds.train_on_sampler(args.train_n, rng_train)

    # ---------------- Eval data (GT ON + OFF) with fixed seed ----------------
    rng_eval = np.random.default_rng(args.eval_seed)
    x_on = ds.eval_on_sampler(args.eval_on_n, rng_eval)
    x_off = ds.eval_off_sampler(args.eval_off_n, rng_eval)
    x_eval = np.concatenate([x_on, x_off], axis=0).astype(np.float32)

    # ---------------- Train & Eval models ----------------
    results = {}

    def run_ae():
        model = AutoEncoder(in_dim=ds.dim, latent_dim=latent_dim).to(device)
        train_ae(model, x_train_t, cfg, device)

        # after training AE model ...
        def proj(xt: torch.Tensor) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                return model(xt.to(device))

        thr = estimate_threshold(
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_val_on=x_val_on,
            threshold_method="percentile",
            threshold_q=args.threshold_q,
            device=device,
        )

        res, cache = eval_with_threshold(
            model_name=f"AE/{ds.name}/z{latent_dim}",
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_test_on=x_on,
            x_test_off=x_off,
            thr=thr,
            device=device,
        )

        results["ae"] = {"res": res, "cache": cache, "model": model, "project_fn": proj}

        # latent scatter
        model.eval()
        with torch.no_grad():
            model.encode(to_tensor(x_eval, device)).cpu().numpy()

        # latent sampling -> decode
        # sample around encoded distribution (mean/cov) to avoid wild regions
        z_train = model.encode(to_tensor(x_train, device)).detach().cpu().numpy()
        if latent_dim == 1:
            mu = float(np.mean(z_train[:, 0]))
            sd = float(np.std(z_train[:, 0]) + 1e-6)
            np.random.default_rng(args.eval_seed + 7).normal(
                mu, sd, size=(args.eval_on_n, 1)
            ).astype(np.float32)
        else:
            mu = np.mean(z_train, axis=0)
            cov = np.cov(z_train.T) + 1e-6 * np.eye(latent_dim)
            np.random.default_rng(args.eval_seed + 7).multivariate_normal(
                mu, cov, size=args.eval_on_n
            ).astype(np.float32)

    def run_vae():
        model = VAE(in_dim=ds.dim, latent_dim=latent_dim).to(device)
        train_vae(model, x_train_t, cfg, device)

        # projection function uses mu (deterministic)
        def proj(x: torch.Tensor) -> torch.Tensor:
            model.eval()
            with torch.no_grad():
                mu, logvar = model.encode(x.to(device))
                x_hat = model.decode(mu)
                return x_hat

        # --- NEW: estimate threshold from val_on (ONLY on-manifold) ---
        thr = estimate_threshold(
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_val_on=x_val_on,
            threshold_method="percentile",
            threshold_q=args.threshold_q,
            device=device,
        )

        # --- NEW: evaluate on test_on + test_off using FIXED thr ---
        res, cache = eval_with_threshold(
            model_name=f"VAE/{ds.name}/z{latent_dim}",
            project_fn=lambda xt: proj(xt),
            errors_fn=lambda a, b: recon_error_l2(a, b),
            x_test_on=x_on,  # 你这里的 x_on / x_off 看起来就是 test set
            x_test_off=x_off,
            thr=thr,
            device=device,
        )

        results["vae"] = {"res": res, "cache": cache, "model": model, "project_fn": proj}

        # latent scatter (use mu)
        model.eval()
        with torch.no_grad():
            model.encode(to_tensor(x_eval, device))

        # latent sampling -> decode
        # For VAE we can sample from standard normal (prior)
        np.random.default_rng(args.eval_seed + 9).normal(size=(args.eval_on_n, latent_dim)).astype(np.float32)

    if args.models in ("ae", "both"):
        run_ae()
    if args.models in ("vae", "both"):
        run_vae()

    ae_pack = None
    vae_pack = None

    # --- AE ---
    if "ae" in results:
        cache = results["ae"]["cache"]
        model = results["ae"]["model"]
        with torch.no_grad():
            z_eval = model.encode(to_tensor(cache["x"], device)).detach().cpu().numpy()
            # latent sample decode（用之前生成的）
            z_train = model.encode(x_train_t).detach().cpu().numpy()
            mu = np.mean(z_train, axis=0)
            cov = np.cov(z_train.T) + 1e-6 * np.eye(latent_dim)
            z_samp = np.random.multivariate_normal(mu, cov, size=args.sample_latent_n).astype(np.float32)
            z_samp_t = to_tensor(z_samp, device)
            x_dec = model.decode(z_samp_t).detach().cpu().numpy()
        ae_pack = (cache, z_eval, x_dec, z_samp)

    # --- VAE ---
    if "vae" in results:
        cache = results["vae"]["cache"]
        model = results["vae"]["model"]
        with torch.no_grad():
            mu, _ = model.encode(torch.tensor(cache["x"], dtype=torch.float32, device=device))
            z_eval = mu.detach().cpu().numpy()
            z_samp = np.random.randn(args.sample_latent_n, latent_dim).astype(np.float32)
            x_dec = model.decode(to_tensor(z_samp, device)).detach().cpu().numpy()
        vae_pack = (cache, z_eval, x_dec, z_samp)

    visualize_all(ds, x_train, ae_pack, vae_pack, latent_dim)
    if args.planner == "on":
        rng_plan = np.random.default_rng(args.eval_seed + 999)
        x_pairs = []
        needed_pairs = max(4, args.planner_pairs)
        needed = needed_pairs * 2
        x_samples = ds.eval_on_sampler(needed, rng_plan)
        for i in range(0, needed, 2):
            x_pairs.append((x_samples[i], x_samples[i + 1]))
        x_pairs = x_pairs[:4]

        planner_keys = []
        if args.planner_model == "auto":
            if "ae" in results:
                planner_keys.append("ae")
            if "vae" in results:
                planner_keys.append("vae")
            if not planner_keys:
                raise RuntimeError("Planner requires at least one trained model.")
        else:
            if args.planner_model not in results:
                raise RuntimeError(
                    f"Planner model '{args.planner_model}' not trained; run with --models {args.planner_model}."
                )
            planner_keys = [args.planner_model]

        for planner_key in planner_keys:
            planner_model = results[planner_key]["model"]
            planner_thr = results[planner_key]["res"].threshold

            planner_model.eval()
            if planner_key == "ae":
                encode_fn = lambda x: planner_model.encode(x)
                decode_fn = lambda z: planner_model.decode(z)
                project_fn = lambda x: planner_model(x)
            else:
                encode_fn = lambda x: planner_model.encode(x)[0]
                decode_fn = lambda z: planner_model.decode(z)
                project_fn = lambda x: planner_model.decode(planner_model.encode(x)[0])

            cases = build_planner_cases(
                x_pairs=x_pairs,
                project_fn=project_fn,
                encode_fn=encode_fn,
                decode_fn=decode_fn,
                gt_distance_fn=ds.gt_distance_fn,
                threshold=planner_thr,
                device=device,
                n_steps=args.planner_steps,
            )
            plot_planner_grid(ds, x_train, cases, title_prefix=f"Planner ({planner_key})")

    print("Done.")


if __name__ == "__main__":
    main()
