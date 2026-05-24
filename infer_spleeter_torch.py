#!/usr/bin/env python
"""PyTorch inference entrypoint for the 5-stem Carnatic Spleeter port.

Examples:
    # One file:
    python infer_spleeter_torch.py \\
        --model_dir trained_models/5stems_carnatic_pt \\
        --input audio.wav \\
        --output separated/

    # Directory (recursive):
    python infer_spleeter_torch.py \\
        --model_dir trained_models/5stems_carnatic_pt \\
        --input_dir clips/ \\
        --recursive \\
        --output separated/

Inference is always chunked (default 30s) with a frame_length-sample overlap and
linear crossfade. This gives a single, predictable code path for any input length.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import subprocess
import sys
from contextlib import nullcontext
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from torch import Tensor

from ragaldm.spleeter_torch import STFTProcessor, Stem5UNet

logger = logging.getLogger("infer_spleeter_torch")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 5-stem Spleeter inference (PyTorch).")
    parser.add_argument("--model_dir", type=str, required=True, help="Dir containing params.json + latest.pt (or ckpt_step_*.pt).")
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=str, nargs="+", help="Input audio file(s).")
    input_group.add_argument("--input_dir", type=str, help="Directory of audio files.")
    parser.add_argument("--output", type=str, required=True, help="Output directory.")
    parser.add_argument(
        "--filename_format",
        type=str,
        default="{filename}/{instrument}.{codec}",
        help="Output path template (relative to --output).",
    )
    parser.add_argument("--codec", type=str, default="wav", choices=["wav", "flac", "mp3", "ogg", "m4a"])
    parser.add_argument("--bitrate", type=str, default="192k", help="Used for lossy codecs.")
    parser.add_argument("--offset", type=float, default=0.0, help="Start offset in seconds.")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process (seconds).")
    parser.add_argument("--recursive", action="store_true", help="Recurse into --input_dir.")
    parser.add_argument(
        "--extensions",
        type=str,
        nargs="+",
        default=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
    )
    parser.add_argument("--chunk_seconds", type=float, default=30.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16 autocast (use fp32).")
    return parser.parse_args()


def find_audio_files(directory: str, extensions: List[str], recursive: bool) -> List[str]:
    found: List[str] = []
    if recursive:
        for ext in extensions:
            found.extend(glob(os.path.join(directory, "**", f"*{ext}"), recursive=True))
    else:
        for ext in extensions:
            found.extend(glob(os.path.join(directory, f"*{ext}")))
    return sorted(found)


def _ckpt_step_key(path: Path) -> int:
    match = re.search(r"ckpt_step_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_checkpoint(model_dir: Path) -> Path:
    latest = model_dir / "latest.pt"
    if latest.exists():
        return latest
    candidates = list(model_dir.glob("ckpt_step_*.pt"))
    if candidates:
        return max(candidates, key=_ckpt_step_key)
    raise FileNotFoundError(f"No checkpoint found in {model_dir}")


def load_model(model_dir: Path, device: torch.device) -> tuple[Stem5UNet, Dict]:
    params_path = model_dir / "params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing params.json at {params_path}")
    with open(params_path) as f:
        params = json.load(f)

    ckpt_path = find_checkpoint(model_dir)
    logger.info("Loading checkpoint: %s", ckpt_path)
    bundle = torch.load(ckpt_path, map_location=device, weights_only=False)

    model_block = params.get("model", {})
    model = Stem5UNet(
        instruments=params["instrument_list"],
        n_channels=params["n_channels"],
        conv_n_filters=params.get("conv_n_filters", [16, 32, 64, 128, 256, 512]),
        conv_activation=model_block.get("params", {}).get("conv_activation", "ELU"),
        deconv_activation=model_block.get("params", {}).get("deconv_activation", "ELU"),
        separation_exponent=params.get("separation_exponent", 2),
        model_type=model_block.get("type", "unet.unet"),
    ).to(device).eval()
    model.load_state_dict(bundle["state_dict"])
    return model, params


def _load_waveform(path: str, target_sr: int, offset: float, duration: Optional[float]) -> np.ndarray:
    """Returns (n_channels, n_samples) float32."""
    info = sf.info(path)
    start = int(offset * info.samplerate)
    frames = -1 if duration is None else int(duration * info.samplerate)
    wav, file_sr = sf.read(path, start=start, frames=frames, dtype="float32", always_2d=True)
    wav = wav.T  # (n_channels, n_samples)
    if file_sr != target_sr:
        wav = librosa.resample(wav, orig_sr=file_sr, target_sr=target_sr, res_type="kaiser_fast")
    return wav.astype(np.float32, copy=False)


def _coerce_channels(wav: np.ndarray, target_ch: int) -> np.ndarray:
    cur_ch = wav.shape[0]
    if cur_ch == target_ch:
        return wav
    if cur_ch == 1 and target_ch == 2:
        return np.tile(wav, (2, 1))
    if cur_ch == 2 and target_ch == 1:
        return wav.mean(axis=0, keepdims=True)
    if cur_ch > target_ch:
        return wav[:target_ch]
    pad = np.zeros((target_ch - cur_ch, wav.shape[1]), dtype=wav.dtype)
    return np.concatenate([wav, pad], axis=0)


@torch.no_grad()
def _separate_chunk(
    chunk_wav: Tensor,
    model: Stem5UNet,
    stft_proc: STFTProcessor,
    use_bf16: bool,
    device: torch.device,
) -> Dict[str, Tensor]:
    """Run a single chunk through the model. Returns {stem: (n_channels, n_samples)}."""
    n_samples = chunk_wav.shape[-1]
    stft = stft_proc.stft(chunk_wav)                 # (n_channels, n_frames, n_freq) complex
    mag = stft_proc.magnitude(stft)
    mag_cropped = stft_proc.crop_freq(mag)           # (n_channels, n_frames, F)
    n_frames = mag_cropped.shape[1]
    segments = stft_proc.pad_and_partition(mag_cropped)  # (n_segments, n_channels, T, F)

    autocast_ctx = (
        torch.autocast(device_type=device.type, dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )
    with autocast_ctx:
        outputs = model(segments)                    # dict[stem -> (n_segments, n_channels, T, F)]

    # compute_masks runs in fp32 inside the model.
    masks_1024 = model.compute_masks(outputs)
    out_waveforms: Dict[str, Tensor] = {}
    for stem, mask_seg in masks_1024.items():
        mask_full = stft_proc.extend_mask_zeros(mask_seg)            # (n_segments, n_channels, T, n_freq)
        mask_full = stft_proc.reassemble_segments(mask_full, n_frames)  # (n_channels, n_frames, n_freq)
        masked_stft = stft * mask_full.to(stft.dtype)
        wav = stft_proc.istft(masked_stft, time_crop=n_samples)
        out_waveforms[stem] = wav.cpu()
    return out_waveforms


def _separate_full_waveform(
    waveform: np.ndarray,
    model: Stem5UNet,
    stft_proc: STFTProcessor,
    sample_rate: int,
    chunk_seconds: float,
    use_bf16: bool,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Chunk + overlap-add separation. Returns {stem: (n_channels, n_samples)} on CPU."""
    n_channels, n_samples = waveform.shape
    chunk_samples = int(chunk_seconds * sample_rate)
    overlap = stft_proc.frame_length
    if chunk_samples <= 2 * overlap:
        chunk_samples = 4 * overlap
    hop = chunk_samples - overlap

    instruments = model.instruments
    outputs: Dict[str, np.ndarray] = {
        stem: np.zeros((n_channels, n_samples), dtype=np.float32) for stem in instruments
    }
    weights = np.zeros(n_samples, dtype=np.float32)

    # Triangular crossfade window.
    win = np.ones(chunk_samples, dtype=np.float32)
    if overlap > 0:
        ramp = np.linspace(0.0, 1.0, overlap, dtype=np.float32, endpoint=False)
        win[:overlap] = ramp
        win[-overlap:] = ramp[::-1]

    waveform_t = torch.from_numpy(waveform)
    start = 0
    n_chunks = max(1, math.ceil(max(n_samples - overlap, 1) / hop))
    for i in range(n_chunks):
        end = min(start + chunk_samples, n_samples)
        chunk = waveform_t[:, start:end]
        if chunk.shape[-1] < chunk_samples:
            pad_amount = chunk_samples - chunk.shape[-1]
            chunk = torch.nn.functional.pad(chunk, (0, pad_amount))
        chunk = chunk.to(device, non_blocking=True)
        stem_waves = _separate_chunk(chunk, model, stft_proc, use_bf16, device)
        actual_len = end - start
        # Adjust crossfade window for short final chunk.
        chunk_win = win.copy()
        if actual_len < chunk_samples:
            chunk_win[actual_len:] = 0.0
        for stem in instruments:
            piece = stem_waves[stem].numpy()[:, :actual_len].astype(np.float32, copy=False)
            outputs[stem][:, start:start + actual_len] += piece * chunk_win[:actual_len]
        weights[start:start + actual_len] += chunk_win[:actual_len]
        if end >= n_samples:
            break
        start += hop

    weights = np.maximum(weights, 1e-8)
    for stem in instruments:
        outputs[stem] /= weights[None, :]
    return outputs


def _write_stem(path: Path, wav: np.ndarray, sample_rate: int, codec: str, bitrate: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # soundfile expects (n_samples, n_channels).
    interleaved = wav.T.astype(np.float32, copy=False)
    if codec in ("wav", "flac"):
        sf.write(str(path), interleaved, sample_rate, subtype="PCM_16" if codec == "wav" else "PCM_24")
        return
    # Lossy codecs via ffmpeg.
    cmd = [
        "ffmpeg", "-y",
        "-f", "f32le",
        "-ar", str(sample_rate),
        "-ac", str(wav.shape[0]),
        "-i", "pipe:0",
        "-b:a", bitrate,
        str(path),
    ]
    proc = subprocess.run(cmd, input=interleaved.tobytes(), check=False, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {path}: {proc.stderr.decode(errors='replace')[:200]}")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    use_bf16 = not args.no_bf16 and device.type == "cuda"

    model_dir = Path(args.model_dir)
    model, params = load_model(model_dir, device)
    sample_rate = int(params["sample_rate"])
    stft_proc = STFTProcessor(
        frame_length=params["frame_length"],
        frame_step=params["frame_step"],
        n_channels=params["n_channels"],
        T=params["T"],
        F=params["F"],
        device=device,
    )

    if args.input:
        files = list(args.input)
    else:
        files = find_audio_files(args.input_dir, args.extensions, args.recursive)
        if not files:
            logger.error("No audio files found under %s", args.input_dir)
            return 1
    logger.info("Processing %d file(s).", len(files))

    output_root = Path(args.output)
    for i, file_path in enumerate(files, 1):
        logger.info("[%d/%d] %s", i, len(files), file_path)
        try:
            wav = _load_waveform(file_path, sample_rate, args.offset, args.duration)
            wav = _coerce_channels(wav, params["n_channels"])
            stems = _separate_full_waveform(
                wav, model, stft_proc, sample_rate, args.chunk_seconds, use_bf16, device
            )
            base = Path(file_path).stem
            for stem_name, stem_wav in stems.items():
                rel = args.filename_format.format(filename=base, instrument=stem_name, codec=args.codec)
                _write_stem(output_root / rel, stem_wav, sample_rate, args.codec, args.bitrate)
        except Exception as e:  # noqa: BLE001
            logger.error("Failed on %s: %s", file_path, e)
            continue
    logger.info("Done. Output: %s", output_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
