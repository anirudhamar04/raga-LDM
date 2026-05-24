#!/usr/bin/env python
"""Convert a Deezer Spleeter TF checkpoint to a PyTorch .pt for the 5-stem PyTorch port.

Workflow:
    # First run: inspect TF checkpoint to confirm naming layout.
    python convert_spleeter_weights.py \\
        --pretrained spleeter:5stems \\
        --target_config legacy/spleeter/configs/5stems_carnatic/base_config.json \\
        --dry_run

    # Then run for real:
    python convert_spleeter_weights.py \\
        --pretrained spleeter:5stems \\
        --target_config legacy/spleeter/configs/5stems_carnatic/base_config.json \\
        --output trained_models/pretrained_5stems_carnatic.pt

Requires `tensorflow==2.15` (read the source TF checkpoint). The pretrained weights
themselves are downloaded directly from the Deezer GitHub release over HTTPS — the
`spleeter` package itself is NOT a runtime dep. The PyTorch model and training/inference
scripts have no TF dependency.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("convert_spleeter_weights")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Mirrors spleeter.model.provider.github.GithubModelProvider defaults.
_GITHUB_HOST = os.environ.get("SPLEETER_GITHUB_HOST", "https://github.com")
_GITHUB_REPO = os.environ.get("SPLEETER_GITHUB_REPO", "deezer/spleeter")
_GITHUB_RELEASE = os.environ.get("SPLEETER_GITHUB_RELEASE", "v1.4.0")

# Spleeter pretrained checkpoints use the *global* Keras auto-incrementing counter
# with no per-instrument scope. Variables look like `conv2d_N/kernel`,
# `conv2d_transpose_N/kernel`, `batch_normalization_N/gamma`, where N is the global
# construction-order index. We recover per-stem grouping by slicing by N:
#   - conv2d:             7 per stem (6 encoder + 1 final 4x4)
#   - conv2d_transpose:   6 per stem (decoder)
#   - batch_normalization: 12 per stem (6 encoder + 6 decoder)
_LAYER_RE = re.compile(
    r"^(?P<layer>conv2d_transpose|batch_normalization|conv2d)(?:_(?P<idx>\d+))?/"
    r"(?P<param>kernel|bias|gamma|beta|moving_mean|moving_variance)$"
)
_CONV2D_PER_STEM = 7
_CONV2D_TRANSPOSE_PER_STEM = 6
_BN_PER_STEM = 12


def _default_cache_dir() -> Path:
    base = os.environ.get("MODEL_PATH") or os.environ.get("SPLEETER_MODEL_PATH")
    if base:
        return Path(base)
    return Path.home() / "pretrained_models"


def _compute_sha256(path: Path) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _download_pretrained(name: str, dest: Path) -> None:
    """Download deezer/spleeter release tarball and extract into dest."""
    import httpx

    base = f"{_GITHUB_HOST}/{_GITHUB_REPO}/releases/download/{_GITHUB_RELEASE}"
    archive_url = f"{base}/{name}.tar.gz"
    checksum_url = f"{base}/checksum.json"

    # httpx <0.22 follows redirects by default; >=0.22 disables them. Construct kwargs
    # dynamically so this script works on both old and new httpx pinned by the lockfile.
    import inspect
    client_kwargs = {"http2": True, "timeout": 60.0}
    if "follow_redirects" in inspect.signature(httpx.Client.__init__).parameters:
        client_kwargs["follow_redirects"] = True

    logger.info("Fetching checksum index from %s", checksum_url)
    with httpx.Client(**client_kwargs) as client:
        index_resp = client.get(checksum_url)
        index_resp.raise_for_status()
        expected_checksum = index_resp.json().get(name)
        if expected_checksum is None:
            raise ValueError(f"No checksum found for model '{name}' in the release index.")

        logger.info("Downloading %s", archive_url)
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            with client.stream("GET", archive_url) as response:
                response.raise_for_status()
                for chunk in response.iter_raw():
                    tmp.write(chunk)
    try:
        actual = _compute_sha256(tmp_path)
        if actual != expected_checksum:
            raise IOError(
                f"Checksum mismatch for {name}.tar.gz: got {actual}, expected {expected_checksum}"
            )
        dest.mkdir(parents=True, exist_ok=True)
        logger.info("Extracting %s -> %s", tmp_path, dest)
        with tarfile.open(tmp_path) as tar:
            tar.extractall(dest)
    finally:
        try:
            tmp_path.unlink()
        except OSError:
            pass


def _get_pretrained_tf_dir(pretrained: str) -> str:
    """Return the on-disk path of the pretrained TF checkpoint, downloading if needed."""
    name = pretrained.split(":", 1)[1] if ":" in pretrained else pretrained
    cache = _default_cache_dir() / name
    if not cache.exists() or not any(cache.iterdir()):
        logger.info("Pretrained model not cached; downloading '%s'.", name)
        _download_pretrained(name, cache)
    else:
        logger.info("Using cached pretrained model: %s", cache)
    return str(cache)


def list_checkpoint_variables(ckpt_dir: str) -> List[Tuple[str, Tuple[int, ...]]]:
    """Return sorted [(var_name, shape), ...] from the latest checkpoint in ckpt_dir."""
    import tensorflow as tf  # type: ignore

    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    logger.info(f"Reading checkpoint: {latest}")
    reader = tf.train.load_checkpoint(latest)
    shape_map = reader.get_variable_to_shape_map()
    return sorted((name, tuple(shape)) for name, shape in shape_map.items())


def _read_tf_vars(ckpt_dir: str) -> Dict[str, np.ndarray]:
    import tensorflow as tf  # type: ignore

    latest = tf.train.latest_checkpoint(ckpt_dir)
    if latest is None:
        raise FileNotFoundError(f"No checkpoint found under {ckpt_dir}")
    reader = tf.train.load_checkpoint(latest)
    return {name: reader.get_tensor(name) for name in reader.get_variable_to_shape_map()}


def _parse_var_name(name: str) -> Optional[Dict[str, object]]:
    """Parse `{layer}[_{idx}]/{param}`.

    Returns dict with keys layer (str), idx (int), param (str). None if no match.
    Strips an optional leading `:0` suffix.
    """
    bare = name.split(":", 1)[0]
    match = _LAYER_RE.match(bare)
    if match is None:
        return None
    idx = int(match.group("idx")) if match.group("idx") else 0
    return {
        "layer": match.group("layer"),
        "idx": idx,
        "param": match.group("param"),
    }


def _group_vars_by_source_stem(
    tf_vars: Dict[str, np.ndarray],
    pretrained_stems: List[str],
) -> Dict[str, Dict[Tuple[str, int, str], np.ndarray]]:
    """Bucket TF variables by *source stem* using construction-order slicing.

    For each layer type, sort by global index then slice by (CONV2D_PER_STEM /
    CONV2D_TRANSPOSE_PER_STEM / BN_PER_STEM); the i-th slice belongs to
    pretrained_stems[i]. Within each slice we re-index to per-stem local indices
    (0..N-1), so downstream code can address `(layer, local_idx, param)`.
    """
    by_layer: Dict[str, Dict[int, Dict[str, np.ndarray]]] = {}
    for name, arr in tf_vars.items():
        lower = name.lower()
        if any(s in lower for s in ("adam", "global_step", "beta1_power", "beta2_power")):
            continue
        parsed = _parse_var_name(name)
        if parsed is None:
            continue
        layer = str(parsed["layer"])
        idx = int(parsed["idx"])
        param = str(parsed["param"])
        by_layer.setdefault(layer, {}).setdefault(idx, {})[param] = arr

    per_stride = {
        "conv2d": _CONV2D_PER_STEM,
        "conv2d_transpose": _CONV2D_TRANSPOSE_PER_STEM,
        "batch_normalization": _BN_PER_STEM,
    }
    expected_total = {layer: per_stride[layer] * len(pretrained_stems) for layer in per_stride}
    grouped: Dict[str, Dict[Tuple[str, int, str], np.ndarray]] = {s: {} for s in pretrained_stems}
    for layer, stride in per_stride.items():
        idx_map = by_layer.get(layer, {})
        ordered = sorted(idx_map.keys())
        if len(ordered) != expected_total[layer]:
            raise RuntimeError(
                f"TF checkpoint has {len(ordered)} '{layer}' layers; expected "
                f"{expected_total[layer]} ({stride} per stem x {len(pretrained_stems)} stems)."
            )
        # Verify the indices are contiguous 0..N-1 (no gaps).
        if ordered != list(range(expected_total[layer])):
            raise RuntimeError(
                f"Non-contiguous '{layer}' indices in TF checkpoint: {ordered[:10]}..."
            )
        for stem_pos, source_stem in enumerate(pretrained_stems):
            start = stem_pos * stride
            for local_idx in range(stride):
                global_idx = start + local_idx
                params = idx_map[global_idx]
                for param_name, arr in params.items():
                    grouped[source_stem][(layer, local_idx, param_name)] = arr
    return grouped


def create_stem_mapping(
    pretrained_stems: List[str], target_stems: List[str]
) -> Dict[str, str]:
    """Map each target stem to a pretrained source stem.

    Strategy (adapted from legacy/utils/init_pretrained_weights.py with the dead-code
    branch removed):
      - vocals -> vocals if both have it.
      - drums -> mridangam AND ghatam (both percussion targets reuse the same source).
      - piano -> violin (preferred), then other -> violin, then bass -> violin.
      - bass -> drone (preferred), then other -> drone.
    """
    mapping: Dict[str, str] = {}
    if "vocals" in pretrained_stems and "vocals" in target_stems:
        mapping["vocals"] = "vocals"
    if "drums" in pretrained_stems:
        if "mridangam" in target_stems:
            mapping["mridangam"] = "drums"
        if "ghatam" in target_stems:
            mapping["ghatam"] = "drums"
    if "violin" in target_stems:
        if "piano" in pretrained_stems:
            mapping["violin"] = "piano"
        elif "other" in pretrained_stems:
            mapping["violin"] = "other"
        elif "bass" in pretrained_stems:
            mapping["violin"] = "bass"
    if "drone" in target_stems:
        if "bass" in pretrained_stems:
            mapping["drone"] = "bass"
        elif "other" in pretrained_stems:
            mapping["drone"] = "other"
    return mapping


def _infer_pretrained_stems(pretrained: str) -> List[str]:
    if "4stems" in pretrained:
        return ["vocals", "drums", "bass", "other"]
    if "5stems" in pretrained:
        return ["vocals", "piano", "drums", "bass", "other"]
    if "2stems" in pretrained:
        return ["vocals", "accompaniment"]
    raise ValueError(f"Cannot infer pretrained stems from name: {pretrained}")


# Layout transposes.
def _transpose_conv2d_kernel(arr: np.ndarray) -> np.ndarray:
    # TF Conv2D: (H, W, in, out) -> PyTorch Conv2d: (out, in, H, W).
    return np.transpose(arr, (3, 2, 0, 1)).copy()


def _transpose_conv2d_transpose_kernel(arr: np.ndarray) -> np.ndarray:
    # TF Conv2DTranspose: (H, W, out, in) -> PyTorch ConvTranspose2d: (in, out, H, W).
    return np.transpose(arr, (3, 2, 0, 1)).copy()


def _load_unet_weights_into_state_dict(
    target_state: Dict[str, np.ndarray],
    target_stem: str,
    source_vars: Dict[Tuple[str, int, str], np.ndarray],
) -> Tuple[List[str], List[str]]:
    """Write the source TF U-Net weights into target_state under unets.{target_stem}.*

    Returns (filled_keys, unmapped_source_keys).
    """
    prefix = f"unets.{target_stem}."
    filled: List[str] = []
    used: set = set()

    def take(layer: str, idx: int, param: str) -> Optional[np.ndarray]:
        key = (layer, idx, param)
        arr = source_vars.get(key)
        if arr is not None:
            used.add(key)
        return arr

    # Encoder convs: TF (conv2d, 0..5) -> PT enc_convs[0..5].
    for i in range(6):
        w = take("conv2d", i, "kernel")
        b = take("conv2d", i, "bias")
        if w is None or b is None:
            raise ValueError(f"Missing TF enc_conv {i} for stem {target_stem}")
        target_state[f"{prefix}enc_convs.{i}.weight"] = _transpose_conv2d_kernel(w)
        target_state[f"{prefix}enc_convs.{i}.bias"] = b.copy()
        filled.extend([f"{prefix}enc_convs.{i}.weight", f"{prefix}enc_convs.{i}.bias"])

    # Encoder BNs: TF (batch_normalization, 0..5) -> PT enc_bns[0..5].
    for i in range(6):
        gamma = take("batch_normalization", i, "gamma")
        beta = take("batch_normalization", i, "beta")
        mean = take("batch_normalization", i, "moving_mean")
        var = take("batch_normalization", i, "moving_variance")
        if any(x is None for x in (gamma, beta, mean, var)):
            raise ValueError(f"Missing TF enc_bn {i} for stem {target_stem}")
        target_state[f"{prefix}enc_bns.{i}.weight"] = gamma.copy()
        target_state[f"{prefix}enc_bns.{i}.bias"] = beta.copy()
        target_state[f"{prefix}enc_bns.{i}.running_mean"] = mean.copy()
        target_state[f"{prefix}enc_bns.{i}.running_var"] = var.copy()
        target_state[f"{prefix}enc_bns.{i}.num_batches_tracked"] = np.array(0, dtype=np.int64)
        filled.extend(
            [
                f"{prefix}enc_bns.{i}.weight",
                f"{prefix}enc_bns.{i}.bias",
                f"{prefix}enc_bns.{i}.running_mean",
                f"{prefix}enc_bns.{i}.running_var",
                f"{prefix}enc_bns.{i}.num_batches_tracked",
            ]
        )

    # Decoder ConvTransposes: TF (conv2d_transpose, 0..5) -> PT dec_convs[0..5].
    for i in range(6):
        w = take("conv2d_transpose", i, "kernel")
        b = take("conv2d_transpose", i, "bias")
        if w is None or b is None:
            raise ValueError(f"Missing TF dec_conv {i} for stem {target_stem}")
        target_state[f"{prefix}dec_convs.{i}.weight"] = _transpose_conv2d_transpose_kernel(w)
        target_state[f"{prefix}dec_convs.{i}.bias"] = b.copy()
        filled.extend([f"{prefix}dec_convs.{i}.weight", f"{prefix}dec_convs.{i}.bias"])

    # Decoder BNs: TF (batch_normalization, 6..11) -> PT dec_bns[0..5].
    for i in range(6):
        tf_idx = i + 6
        gamma = take("batch_normalization", tf_idx, "gamma")
        beta = take("batch_normalization", tf_idx, "beta")
        mean = take("batch_normalization", tf_idx, "moving_mean")
        var = take("batch_normalization", tf_idx, "moving_variance")
        if any(x is None for x in (gamma, beta, mean, var)):
            raise ValueError(f"Missing TF dec_bn {tf_idx} for stem {target_stem}")
        target_state[f"{prefix}dec_bns.{i}.weight"] = gamma.copy()
        target_state[f"{prefix}dec_bns.{i}.bias"] = beta.copy()
        target_state[f"{prefix}dec_bns.{i}.running_mean"] = mean.copy()
        target_state[f"{prefix}dec_bns.{i}.running_var"] = var.copy()
        target_state[f"{prefix}dec_bns.{i}.num_batches_tracked"] = np.array(0, dtype=np.int64)
        filled.extend(
            [
                f"{prefix}dec_bns.{i}.weight",
                f"{prefix}dec_bns.{i}.bias",
                f"{prefix}dec_bns.{i}.running_mean",
                f"{prefix}dec_bns.{i}.running_var",
                f"{prefix}dec_bns.{i}.num_batches_tracked",
            ]
        )

    # Final conv: TF (conv2d, 6) -> PT final_conv.
    w = take("conv2d", 6, "kernel")
    b = take("conv2d", 6, "bias")
    if w is None or b is None:
        raise ValueError(f"Missing TF final_conv for stem {target_stem}")
    target_state[f"{prefix}final_conv.weight"] = _transpose_conv2d_kernel(w)
    target_state[f"{prefix}final_conv.bias"] = b.copy()
    filled.extend([f"{prefix}final_conv.weight", f"{prefix}final_conv.bias"])

    unmapped_for_stem = [
        f"{layer}_{idx}/{param}" for (layer, idx, param) in source_vars if (layer, idx, param) not in used
    ]
    return filled, unmapped_for_stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert TF Spleeter weights to PyTorch .pt")
    parser.add_argument(
        "--pretrained",
        type=str,
        default="spleeter:5stems",
        help="Pretrained model name (e.g. spleeter:4stems, spleeter:5stems).",
    )
    parser.add_argument(
        "--target_config",
        type=str,
        required=True,
        help="Path to target model config JSON (e.g. 5stems_carnatic/base_config.json).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output .pt path. Required unless --dry_run is set.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print TF checkpoint variables (name + shape) and exit.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Error if any PyTorch state_dict key is unfilled after conversion.",
    )
    args = parser.parse_args()

    pretrained_dir = _get_pretrained_tf_dir(args.pretrained)
    variables = list_checkpoint_variables(pretrained_dir)

    if args.dry_run:
        print(f"\n# Variables in {args.pretrained} ({len(variables)} total):\n")
        for name, shape in variables:
            print(f"  {name:80s}  shape={shape}")
        return 0

    if not args.output:
        parser.error("--output is required unless --dry_run is set")

    with open(args.target_config) as f:
        target_config = json.load(f)
    target_stems: List[str] = target_config["instrument_list"]
    n_channels: int = target_config["n_channels"]
    conv_n_filters = target_config.get("conv_n_filters", [16, 32, 64, 128, 256, 512])

    pretrained_stems = _infer_pretrained_stems(args.pretrained)
    stem_mapping = create_stem_mapping(pretrained_stems, target_stems)
    missing_targets = [s for s in target_stems if s not in stem_mapping]
    if missing_targets:
        logger.warning(
            "No source stem available for %s; these PyTorch UNets keep random init.",
            missing_targets,
        )
    logger.info("Stem mapping (target <- source): %s", stem_mapping)

    tf_vars = _read_tf_vars(pretrained_dir)
    grouped = _group_vars_by_source_stem(tf_vars, pretrained_stems)
    logger.info("TF source stems (by construction order): %s", pretrained_stems)

    # Build the target state_dict as numpy arrays first; let torch handle conversion.
    target_state: Dict[str, np.ndarray] = {}
    all_filled: List[str] = []
    all_unmapped: List[str] = []
    for target_stem in target_stems:
        source_stem = stem_mapping.get(target_stem)
        if source_stem is None:
            continue
        source_vars = grouped.get(source_stem)
        if source_vars is None:
            logger.warning(
                "Source stem '%s' not found in TF checkpoint; skipping target '%s'",
                source_stem,
                target_stem,
            )
            continue
        filled, unmapped = _load_unet_weights_into_state_dict(
            target_state, target_stem, source_vars
        )
        all_filled.extend(filled)
        all_unmapped.extend(f"{source_stem}::{u}" for u in unmapped)
        logger.info(
            "Mapped %d tensors for target stem '%s' (<- source '%s')",
            len(filled),
            target_stem,
            source_stem,
        )

    # Build the PyTorch model so we can validate keys + shapes.
    import torch  # imported late: avoids pulling torch into --dry_run path
    from ragaldm.spleeter_torch.model import Stem5UNet

    model = Stem5UNet(
        instruments=target_stems,
        n_channels=n_channels,
        conv_n_filters=conv_n_filters,
    )
    pt_state = model.state_dict()
    expected_keys = set(pt_state.keys())
    filled_keys = set(target_state.keys())
    missing_keys = sorted(expected_keys - filled_keys)
    unexpected_keys = sorted(filled_keys - expected_keys)
    if unexpected_keys:
        raise RuntimeError(f"Converter produced keys not in PT model: {unexpected_keys[:5]}")

    # Shape sanity check.
    shape_mismatches = []
    for k, arr in target_state.items():
        pt_shape = tuple(pt_state[k].shape)
        if tuple(arr.shape) != pt_shape:
            shape_mismatches.append((k, tuple(arr.shape), pt_shape))
    if shape_mismatches:
        for k, got, want in shape_mismatches[:10]:
            logger.error("Shape mismatch %s: got %s, want %s", k, got, want)
        raise RuntimeError(f"{len(shape_mismatches)} shape mismatches detected.")

    if missing_keys:
        msg = f"{len(missing_keys)} PyTorch keys unfilled (will use random init): e.g. {missing_keys[:5]}"
        if args.strict:
            raise RuntimeError(msg)
        logger.warning(msg)

    # Load into model and write.
    tensor_state = {k: torch.from_numpy(v) for k, v in target_state.items()}
    incompatible = model.load_state_dict(tensor_state, strict=False)
    logger.info(
        "load_state_dict: %d missing, %d unexpected",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "instruments": target_stems,
            "source_model": args.pretrained,
            "stem_mapping": stem_mapping,
            "conv_n_filters": conv_n_filters,
            "n_channels": n_channels,
            "missing_keys": list(incompatible.missing_keys),
            "unmapped_tf_vars_count": len(all_unmapped),
        },
        output_path,
    )
    logger.info("Wrote %s (%d tensors).", output_path, len(model.state_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
