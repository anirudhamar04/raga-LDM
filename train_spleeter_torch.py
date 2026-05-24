#!/usr/bin/env python
"""PyTorch training entrypoint for the 5-stem Carnatic Spleeter port.

Training is epoch-based:
- One epoch = one full pass over the training DataLoader.
- After each epoch, validation runs and three checkpoint files are written:
    - ckpt_epoch_{N}.pt   (last `--keep_last` of these are kept)
    - latest.pt           (always the most recent epoch; used by --resume and inference)
    - best.pt             (whenever val/loss strictly improves)
- `--resume` picks up latest.pt: model, optimizer, epoch counter, and best_val_loss
  are all restored, so the best-val tracking survives restarts.

Usage:
    python train_spleeter_torch.py \\
        --config configs/colab_5stems_carnatic.json \\
        --data /content \\
        --pretrained_weights trained_models/pretrained_5stems_carnatic.pt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Optional

import torch
import wandb
from torch.utils.data import DataLoader

from ragaldm.spleeter_torch import Stem5UNet, l1_mask_loss, make_loaders

logger = logging.getLogger("train_spleeter_torch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train 5-stem Spleeter (PyTorch).")
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON.")
    parser.add_argument("--data", type=str, required=True, help="Dataset root; CSV paths are resolved relative to this.")
    parser.add_argument("--pretrained_weights", type=str, default=None, help="Path to .pt from convert_spleeter_weights.py.")
    parser.add_argument("--resume", action="store_true", help="Resume from latest.pt in model_dir.")
    parser.add_argument("--validate_only", action="store_true", help="Run a single validation pass and exit.")
    parser.add_argument("--skip_validation", action="store_true", help="Disable in-loop validation (also disables best.pt updates).")
    parser.add_argument("--num_workers", type=int, default=None, help="DataLoader workers (auto: 8 Linux, 4 Windows).")
    parser.add_argument("--max_epochs", type=int, default=None, help="Override config max_epochs.")
    parser.add_argument("--keep_last", type=int, default=5, help="How many ckpt_epoch_*.pt files to retain.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging.")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 autocast (use fp32).")
    return parser.parse_args()


def load_config(path: str, data_dir: str) -> Dict:
    with open(path) as f:
        params = json.load(f)
    for csv_key in ("train_csv", "validation_csv"):
        csv_path = params[csv_key]
        if not os.path.isabs(csv_path):
            params[csv_key] = os.path.join(data_dir, csv_path)
    return params


def _epoch_from_name(path: Path) -> int:
    match = re.search(r"ckpt_epoch_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def save_epoch_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
    best_val_loss: float,
    model_dir: Path,
    params: Dict,
    keep_last: int,
) -> Path:
    payload = {
        "epoch": epoch,
        "step": step,
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
        "params": params,
    }
    path = model_dir / f"ckpt_epoch_{epoch}.pt"
    torch.save(payload, path)
    torch.save(payload, model_dir / "latest.pt")
    # Trim old epoch checkpoints.
    all_ckpts = sorted(model_dir.glob("ckpt_epoch_*.pt"), key=_epoch_from_name)
    for old in all_ckpts[:-keep_last]:
        try:
            old.unlink()
        except OSError:
            pass
    logger.info("Saved epoch checkpoint: %s", path)
    return path


def save_best_checkpoint(
    model: torch.nn.Module,
    epoch: int,
    step: int,
    val_loss: float,
    model_dir: Path,
    params: Dict,
) -> Path:
    payload = {
        "epoch": epoch,
        "step": step,
        "val_loss": val_loss,
        "state_dict": model.state_dict(),
        "params": params,
    }
    path = model_dir / "best.pt"
    torch.save(payload, path)
    logger.info("New best val/loss=%.6f at epoch %d; wrote %s", val_loss, epoch, path)
    return path


def _move_batch(batch, device: torch.device):
    mix, targets = batch
    mix = mix.to(device, non_blocking=True)
    targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
    return mix, targets


@torch.no_grad()
def run_validation(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    step: int,
    use_bf16: bool,
    wandb_run,
) -> Dict[str, float]:
    model.eval()
    total = 0.0
    per_stem_sums: Dict[str, float] = {}
    n_batches = 0
    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    for batch in val_loader:
        mix, targets = _move_batch(batch, device)
        with autocast_ctx:
            preds = model(mix)
            loss, per_stem = l1_mask_loss(preds, targets)
        total += float(loss.detach().float().item())
        for k, v in per_stem.items():
            per_stem_sums[k] = per_stem_sums.get(k, 0.0) + v
        n_batches += 1
    if n_batches == 0:
        logger.warning("Validation loader produced zero batches.")
        return {}
    metrics = {"val/loss": total / n_batches}
    for k, v in per_stem_sums.items():
        metrics[f"val/{k}_loss"] = v / n_batches
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    logger.info("[val @ step %d] %s", step, {k: round(v, 6) for k, v in metrics.items()})
    return metrics


def _log_device_banner(device: torch.device, use_bf16: bool) -> None:
    """Print a short banner so it's obvious whether training will use CUDA."""
    logger.info("=" * 68)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            logger.warning("Requested --device cuda but torch.cuda.is_available() == False.")
            logger.warning("Falling back to CPU is NOT automatic — training will OOM/crash.")
        else:
            idx = device.index if device.index is not None else torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            total_gb = props.total_memory / (1024 ** 3)
            bf16_native = torch.cuda.is_bf16_supported()
            logger.info("Device: CUDA:%d  %s  (%.1f GB, sm_%d%d)",
                        idx, props.name, total_gb, props.major, props.minor)
            logger.info("CUDA toolkit: %s  cuDNN: %s",
                        torch.version.cuda, torch.backends.cudnn.version())
            logger.info("Precision: bf16 autocast=%s (HW native bf16=%s)",
                        use_bf16, bf16_native)
    else:
        logger.info("Device: CPU (no CUDA). Training will be very slow.")
        logger.info("torch.cuda.is_available()=%s", torch.cuda.is_available())
    logger.info("torch=%s", torch.__version__)
    logger.info("=" * 68)


def main() -> int:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device(args.device)
    use_bf16 = not args.no_bf16 and device.type == "cuda"
    _log_device_banner(device, use_bf16)

    params = load_config(args.config, args.data)
    model_dir = Path(params["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    with open(model_dir / "params.json", "w") as f:
        json.dump(params, f, indent=2)

    logger.info("Building Stem5UNet for instruments=%s", params["instrument_list"])
    model_block = params.get("model", {})
    model = Stem5UNet(
        instruments=params["instrument_list"],
        n_channels=params["n_channels"],
        conv_n_filters=params.get("conv_n_filters", [16, 32, 64, 128, 256, 512]),
        conv_activation=model_block.get("params", {}).get("conv_activation", "ELU"),
        deconv_activation=model_block.get("params", {}).get("deconv_activation", "ELU"),
        separation_exponent=params.get("separation_exponent", 2),
        model_type=model_block.get("type", "unet.unet"),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(params["learning_rate"]))

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")
    latest_ckpt = model_dir / "latest.pt"
    if args.resume and latest_ckpt.exists():
        logger.info("Resuming from %s", latest_ckpt)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        logger.info("Resume state: epoch=%d step=%d best_val_loss=%.6f",
                    start_epoch, global_step, best_val_loss)
    elif args.pretrained_weights:
        logger.info("Loading pretrained weights from %s", args.pretrained_weights)
        bundle = torch.load(args.pretrained_weights, map_location=device, weights_only=False)
        incompat = model.load_state_dict(bundle["state_dict"], strict=False)
        if incompat.missing_keys:
            logger.warning("Pretrained load: %d missing keys (sample: %s)",
                           len(incompat.missing_keys), incompat.missing_keys[:3])
        if incompat.unexpected_keys:
            logger.warning("Pretrained load: %d unexpected keys", len(incompat.unexpected_keys))

    train_loader, val_loader = make_loaders(params, args.data, args.num_workers)

    if args.validate_only:
        run_validation(model, val_loader, device, global_step, use_bf16, wandb_run=None)
        return 0

    wandb_run = None
    if not args.no_wandb:
        wandb_run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "RetinalDistill"),
            project=os.environ.get("WANDB_PROJECT", "CarnaticSpeeter"),
            config=params,
            resume="allow",
            id=os.environ.get("WANDB_RUN_ID"),
        )

    autocast_ctx_fn = (
        (lambda: torch.autocast(device_type=device.type, dtype=torch.bfloat16))
        if use_bf16 and device.type == "cuda"
        else (lambda: nullcontext())
    )

    summary_every = int(params.get("save_summary_steps", 10))
    max_epochs = int(args.max_epochs if args.max_epochs is not None
                     else params.get("max_epochs", 100))
    # Optional hard step cap. If set in config (e.g. for short smoke runs), training
    # stops early and still writes the final epoch checkpoint + best.pt.
    max_steps = int(params.get("train_max_steps", 0)) or None

    logger.info(
        "Starting training: max_epochs=%d batch_size=%s lr=%s bf16=%s device=%s",
        max_epochs, params.get("batch_size", 4), params["learning_rate"], use_bf16, device,
    )

    try:
        for epoch in range(start_epoch, max_epochs):
            model.train()
            epoch_loss_sum = 0.0
            n_train_batches = 0
            for batch in train_loader:
                if max_steps is not None and global_step >= max_steps:
                    break
                mix, targets = _move_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                with autocast_ctx_fn():
                    preds = model(mix)
                    loss, per_stem = l1_mask_loss(preds, targets)
                loss.backward()
                optimizer.step()
                global_step += 1
                n_train_batches += 1
                loss_val = float(loss.detach().float().item())
                epoch_loss_sum += loss_val

                if global_step % summary_every == 0:
                    log_payload = {"train/loss": loss_val, "train/epoch": epoch}
                    for k, v in per_stem.items():
                        log_payload[f"train/{k}_loss"] = v
                    if wandb_run is not None:
                        wandb_run.log(log_payload, step=global_step)
                    logger.info(
                        "epoch %d  step %d  loss=%.6f  %s",
                        epoch, global_step, loss_val,
                        {k: round(v, 6) for k, v in per_stem.items()},
                    )

            mean_train_loss = epoch_loss_sum / max(n_train_batches, 1)
            logger.info("epoch %d done  mean_train_loss=%.6f  steps=%d",
                        epoch, mean_train_loss, global_step)
            if wandb_run is not None and n_train_batches > 0:
                wandb_run.log({"train/epoch_mean_loss": mean_train_loss,
                               "train/epoch": epoch}, step=global_step)

            # Validate + save.
            if not args.skip_validation:
                val_metrics = run_validation(
                    model, val_loader, device, global_step, use_bf16, wandb_run,
                )
                val_loss = val_metrics.get("val/loss", float("inf"))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_best_checkpoint(model, epoch, global_step, val_loss, model_dir, params)

            save_epoch_checkpoint(
                model, optimizer, epoch, global_step, best_val_loss,
                model_dir, params, keep_last=args.keep_last,
            )

            if max_steps is not None and global_step >= max_steps:
                logger.info("Hit train_max_steps=%d cap; ending training.", max_steps)
                break

        logger.info("Training complete. best_val_loss=%.6f", best_val_loss)
    except KeyboardInterrupt:
        logger.info("Training interrupted; saving partial checkpoint.")
        save_epoch_checkpoint(
            model, optimizer, locals().get("epoch", start_epoch), global_step,
            best_val_loss, model_dir, params, keep_last=args.keep_last,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()
    return 0


if __name__ == "__main__":
    sys.exit(main())
