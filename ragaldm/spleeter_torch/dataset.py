"""PyTorch Dataset for Spleeter-format CSVs.

CSV columns (paths relative to data_dir):
    mix_path, vocals_path, violin_path, ghatam_path, mridangam_path, drone_path, duration

Train-only augmentations preserve the source-separation invariant `mix = sum(stems)`
by sampling each augmentation parameter ONCE per __getitem__ call and applying it
identically to the mix waveform and every stem waveform. Available augmentations:

- Chunk stitching (`stitch_prob`): with the chosen probability, sample a second
  chunk index from the same song and splice the two windows at a random point
  with a short crossfade. Same splice plan used for mix + all stems.
- Random gain (`gain_db_max`): sample a single gain in [-gain_db_max, +gain_db_max]
  and multiply every waveform by 10**(gain/20).
- Channel swap (`channel_swap_prob`): with the chosen probability, swap L/R on
  every waveform (stereo only).

Augmentations are disabled in val mode regardless of config.
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from ragaldm.spleeter_torch.audio import STFTProcessor

_Mode = Literal["train", "val"]


@dataclass
class _AugmentationPlan:
    """Decisions taken once per __getitem__; applied to every waveform identically."""
    secondary_k: Optional[int]      # second chunk index for stitching, or None
    splice_sample: Optional[int]    # absolute sample index of the splice point
    gain: Optional[float]           # linear gain factor (>0), or None
    swap_channels: bool


class SpleeterDataset(Dataset):
    MARGIN: float = 0.5
    _CROSSFADE_SAMPLES: int = 512  # ~11.6 ms @ 44.1 kHz

    def __init__(
        self,
        csv_path: str,
        data_dir: str,
        instrument_list: List[str],
        mix_name: str = "mix",
        sample_rate: int = 44100,
        chunk_duration: float = 20.0,
        n_chunks_per_song: int = 2,
        chunk_stride_sec: Optional[float] = None,
        frame_length: int = 4096,
        frame_step: int = 1024,
        T: int = 512,
        F: int = 1024,
        n_channels: int = 2,
        mode: _Mode = "train",
        random_seed: int = 0,
        augmentation: Optional[Dict] = None,
    ) -> None:
        """
        If `chunk_stride_sec` is set, each song contributes
            n_chunks = max(1, floor((duration - chunk_duration - 2*MARGIN) / stride) + 1)
        chunks spaced `chunk_stride_sec` apart, overriding `n_chunks_per_song`.
        Long songs contribute proportionally more samples; short songs as few as 1.
        Val mode always uses a single central chunk regardless of these knobs.
        """
        self.data_dir = data_dir
        self.instrument_list = list(instrument_list)
        self.mix_name = mix_name
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.T = T
        self.F = F
        self.n_channels = n_channels
        self.mode = mode
        # Stride-based chunking is train-only; val always uses one central chunk.
        if mode == "val":
            self.n_chunks_per_song = 1
            self.chunk_stride_sec = None
            self.chunk_duration = min(chunk_duration, 12.0)
        else:
            self.n_chunks_per_song = max(1, n_chunks_per_song)
            self.chunk_stride_sec = (
                float(chunk_stride_sec) if chunk_stride_sec and chunk_stride_sec > 0 else None
            )
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

        aug = augmentation or {}
        if mode == "train":
            self.stitch_prob = float(aug.get("stitch_prob", 0.0))
            self.gain_prob = float(aug.get("gain_prob", 0.0))
            self.gain_db_max = float(aug.get("gain_db_max", 0.0))
            self.channel_swap_prob = float(aug.get("channel_swap_prob", 0.0))
        else:
            self.stitch_prob = 0.0
            self.gain_prob = 0.0
            self.gain_db_max = 0.0
            self.channel_swap_prob = 0.0

        self.df = pd.read_csv(csv_path)
        self._validate_columns()

        # Precompute the flat (row_idx, chunk_idx) index. With stride-mode this is
        # variable per row; in n_chunks mode it's a constant n_chunks_per_song.
        self._index: List[Tuple[int, int]] = []
        self._chunks_per_row: List[int] = []
        for row_idx in range(len(self.df)):
            duration = float(self.df.iloc[row_idx]["duration"])
            n_here = self._compute_n_chunks(duration)
            self._chunks_per_row.append(n_here)
            for k in range(n_here):
                self._index.append((row_idx, k))

        self.stft_proc = STFTProcessor(
            frame_length=frame_length,
            frame_step=frame_step,
            n_channels=n_channels,
            T=T,
            F=F,
            device="cpu",
        )

    def _validate_columns(self) -> None:
        required = [f"{self.mix_name}_path"] + [f"{s}_path" for s in self.instrument_list] + ["duration"]
        missing = [c for c in required if c not in self.df.columns]
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

    def _compute_n_chunks(self, duration: float) -> int:
        """How many chunks does a song of given duration contribute?"""
        if self.chunk_stride_sec is None:
            return self.n_chunks_per_song
        available = duration - self.chunk_duration - 2 * self.MARGIN
        if available <= 0:
            return 1
        return max(1, int(available // self.chunk_stride_sec) + 1)

    def __len__(self) -> int:
        return len(self._index)

    def _segment_start(self, duration: float, k: int, n_chunks_here: int) -> float:
        """Chunk-k start in seconds. Two branches:
        - stride mode: start = k * chunk_stride_sec + MARGIN.
        - legacy n_chunks_per_song mode: evenly-spaced offsets across the song.
        """
        if self.chunk_stride_sec is not None:
            return max(k * self.chunk_stride_sec + self.MARGIN, 0.0)
        if n_chunks_here == 1:
            return max(duration / 2.0 - self.chunk_duration / 2.0, 0.0)
        denom = max(n_chunks_here - 1, 1)
        usable = duration - self.chunk_duration - 2 * self.MARGIN
        return max(k * usable / denom + self.MARGIN, 0.0)

    def _load_chunk(self, rel_path: str, start_sec: float, n_samples: int) -> np.ndarray:
        """Load a chunk_duration window and coerce to (n_channels, n_samples)."""
        path = os.path.join(self.data_dir, rel_path)
        info = sf.info(path)
        file_sr = info.samplerate
        start_frame = max(int(start_sec * file_sr), 0)
        if file_sr == self.sample_rate:
            target_frames = n_samples
        else:
            target_frames = int(math.ceil(n_samples * file_sr / self.sample_rate))
        wav, _ = sf.read(path, start=start_frame, frames=target_frames, dtype="float32", always_2d=True)
        if file_sr != self.sample_rate:
            wav = librosa.resample(wav.T, orig_sr=file_sr, target_sr=self.sample_rate, res_type="kaiser_fast").T
        if wav.shape[0] < n_samples:
            wav = np.pad(wav, ((0, n_samples - wav.shape[0]), (0, 0)))
        elif wav.shape[0] > n_samples:
            wav = wav[:n_samples]
        cur_ch = wav.shape[1]
        if cur_ch != self.n_channels:
            if cur_ch == 1 and self.n_channels == 2:
                wav = np.tile(wav, (1, 2))
            elif cur_ch == 2 and self.n_channels == 1:
                wav = wav.mean(axis=1, keepdims=True)
            elif cur_ch > self.n_channels:
                wav = wav[:, : self.n_channels]
            else:
                pad_ch = self.n_channels - cur_ch
                wav = np.concatenate(
                    [wav, np.zeros((wav.shape[0], pad_ch), dtype=wav.dtype)], axis=1
                )
        return wav.T.astype(np.float32, copy=False)  # (n_channels, n_samples)

    def _splice(self, wav_a: np.ndarray, wav_b: np.ndarray, splice_sample: int) -> np.ndarray:
        """Crossfade wav_a[:splice] with wav_b[splice:] over a short window."""
        n = wav_a.shape[-1]
        xf = self._CROSSFADE_SAMPLES
        # If splice point is too close to either edge, fall back to a hard cut.
        if splice_sample <= xf or splice_sample >= n - xf:
            return np.concatenate(
                [wav_a[:, :splice_sample], wav_b[:, splice_sample:]], axis=-1
            )
        out = np.empty_like(wav_a)
        xf_start = splice_sample - xf // 2
        xf_end = xf_start + xf
        out[:, :xf_start] = wav_a[:, :xf_start]
        fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)
        fade_out = 1.0 - fade_in
        out[:, xf_start:xf_end] = (
            wav_a[:, xf_start:xf_end] * fade_out
            + wav_b[:, xf_start:xf_end] * fade_in
        )
        out[:, xf_end:] = wav_b[:, xf_end:]
        return out

    def _build_plan(
        self, k_primary: int, n_chunks_here: int, n_samples: int
    ) -> _AugmentationPlan:
        """Sample augmentation decisions for this __getitem__ call."""
        secondary_k: Optional[int] = None
        splice_sample: Optional[int] = None
        if (
            self.mode == "train"
            and n_chunks_here > 1
            and self.stitch_prob > 0.0
            and self._rng.random() < self.stitch_prob
        ):
            choices = [k for k in range(n_chunks_here) if k != k_primary]
            secondary_k = int(self._rng.choice(choices))
            # Keep the splice safely away from the boundaries so the crossfade fits.
            min_splice = self._CROSSFADE_SAMPLES + 1
            max_splice = n_samples - self._CROSSFADE_SAMPLES - 1
            if max_splice > min_splice:
                splice_sample = int(self._rng.integers(min_splice, max_splice))
            else:
                secondary_k = None  # chunk too short — skip stitching

        gain: Optional[float] = None
        if (
            self.gain_prob > 0.0
            and self.gain_db_max > 0.0
            and self._rng.random() < self.gain_prob
        ):
            db = float(self._rng.uniform(-self.gain_db_max, self.gain_db_max))
            gain = float(10.0 ** (db / 20.0))

        swap_channels = (
            self.n_channels == 2
            and self.channel_swap_prob > 0.0
            and self._rng.random() < self.channel_swap_prob
        )

        return _AugmentationPlan(
            secondary_k=secondary_k,
            splice_sample=splice_sample,
            gain=gain,
            swap_channels=swap_channels,
        )

    def _load_with_plan(
        self,
        rel_path: str,
        duration: float,
        k_primary: int,
        n_chunks_here: int,
        n_samples: int,
        plan: _AugmentationPlan,
    ) -> np.ndarray:
        primary = self._load_chunk(
            rel_path, self._segment_start(duration, k_primary, n_chunks_here), n_samples
        )
        if plan.secondary_k is not None and plan.splice_sample is not None:
            secondary = self._load_chunk(
                rel_path,
                self._segment_start(duration, plan.secondary_k, n_chunks_here),
                n_samples,
            )
            primary = self._splice(primary, secondary, plan.splice_sample)
        if plan.gain is not None:
            primary = primary * plan.gain
        if plan.swap_channels and self.n_channels == 2:
            primary = primary[[1, 0], :]
        return primary

    def _to_input_spec(
        self, waveform_np: np.ndarray, t_start: Optional[int] = None
    ) -> Tuple[Tensor, int]:
        """waveform_np: (n_channels, n_samples) -> ((n_channels, T, F) spec, t_start)."""
        wav = torch.from_numpy(waveform_np)
        stft = self.stft_proc.stft(wav)
        mag = self.stft_proc.magnitude(stft)
        mag = self.stft_proc.crop_freq(mag)
        n_frames = mag.shape[1]
        if n_frames < self.T:
            pad = self.T - n_frames
            mag = torch.nn.functional.pad(mag, (0, 0, 0, pad))
            return mag, 0
        if t_start is None:
            if self.mode == "train":
                max_start = n_frames - self.T
                t_start = int(self._rng.integers(0, max_start + 1))
            else:
                t_start = (n_frames - self.T) // 2
        return mag[:, t_start : t_start + self.T, :], t_start

    def __getitem__(self, idx: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        row_idx, k = self._index[idx]
        row = self.df.iloc[row_idx]
        duration = float(row["duration"])
        n_chunks_here = self._chunks_per_row[row_idx]
        n_samples = int(self.chunk_duration * self.sample_rate)
        plan = self._build_plan(k_primary=k, n_chunks_here=n_chunks_here, n_samples=n_samples)

        mix_wav = self._load_with_plan(
            row[f"{self.mix_name}_path"], duration, k, n_chunks_here, n_samples, plan
        )
        mix_spec, t_start = self._to_input_spec(mix_wav, t_start=None)

        stem_specs: Dict[str, Tensor] = {}
        for inst in self.instrument_list:
            wav = self._load_with_plan(
                row[f"{inst}_path"], duration, k, n_chunks_here, n_samples, plan
            )
            spec, _ = self._to_input_spec(wav, t_start=t_start)
            stem_specs[inst] = spec

        return mix_spec, stem_specs


def _worker_init_fn(worker_id: int) -> None:
    seed = (torch.initial_seed() + worker_id) % (2**32)
    np.random.seed(seed)


def make_loaders(
    params: dict,
    data_dir: str,
    num_workers: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader]:
    """Build train + val DataLoaders from a Spleeter-style params dict."""
    if num_workers is None:
        num_workers = 4 if sys.platform == "win32" else 8

    augmentation = params.get("augmentation")

    def _build(csv_key: str, mode: _Mode) -> Dataset:
        return SpleeterDataset(
            csv_path=params[csv_key],
            data_dir=data_dir,
            instrument_list=params["instrument_list"],
            mix_name=params.get("mix_name", "mix"),
            sample_rate=params["sample_rate"],
            chunk_duration=params.get("chunk_duration", 20.0),
            n_chunks_per_song=params.get("n_chunks_per_song", 2),
            chunk_stride_sec=params.get("chunk_stride_sec"),
            frame_length=params["frame_length"],
            frame_step=params["frame_step"],
            T=params["T"],
            F=params["F"],
            n_channels=params["n_channels"],
            mode=mode,
            random_seed=params.get("random_seed", 42) + (0 if mode == "train" else 1),
            augmentation=augmentation,
        )

    train_ds = _build("train_csv", "train")
    val_ds = _build("validation_csv", "val")

    common_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=_worker_init_fn,
    )
    if sys.platform == "win32" and num_workers > 0:
        common_kwargs["multiprocessing_context"] = "spawn"

    train_loader = DataLoader(
        train_ds,
        batch_size=params.get("batch_size", 4),
        shuffle=True,
        drop_last=True,
        **common_kwargs,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=params.get("batch_size", 4),
        shuffle=False,
        drop_last=False,
        **common_kwargs,
    )
    return train_loader, val_loader
