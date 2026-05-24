"""STFT / iSTFT pipeline byte-equivalent to Deezer Spleeter's TF graph.

Mirrors legacy/spleeter/spleeter/model/__init__.py:_build_stft_feature and _inverse_stft:
- Pre-pads the waveform with frame_length zeros.
- Uses pad_end=True semantics so the trailing frame is fully populated.
- Hann periodic window.
- iSTFT multiplies by 2/3 (Hann 50%-overlap energy compensation) and strips the
  leading frame_length samples.
"""

from __future__ import annotations

import math
from typing import Union

import torch
from torch import Tensor

WINDOW_COMPENSATION_FACTOR: float = 2.0 / 3.0


class STFTProcessor:
    def __init__(
        self,
        frame_length: int = 4096,
        frame_step: int = 1024,
        n_channels: int = 2,
        T: int = 512,
        F: int = 1024,
        device: Union[str, torch.device] = "cpu",
    ) -> None:
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.n_channels = n_channels
        self.T = T
        self.F = F
        self.n_freq = frame_length // 2 + 1
        self.device = torch.device(device)
        self._window = torch.hann_window(frame_length, periodic=True, device=self.device)

    def to(self, device: Union[str, torch.device]) -> "STFTProcessor":
        self.device = torch.device(device)
        self._window = self._window.to(self.device)
        return self

    def _pad_for_stft(self, waveform: Tensor) -> tuple[Tensor, int]:
        """Prepend frame_length zeros and end-pad so the last frame is full.

        Matches tf.signal.stft(..., pad_end=True) plus the explicit leading-frame
        zero pad from Spleeter's _build_stft_feature.

        Args:
            waveform: (n_channels, n_samples) float tensor.

        Returns:
            (padded_waveform, n_frames) where padded_waveform has shape
            (n_channels, padded_samples).
        """
        n_samples = waveform.shape[-1]
        front_pad = self.frame_length
        total = n_samples + front_pad
        if total <= self.frame_length:
            n_frames = 1
        else:
            n_frames = math.ceil((total - self.frame_length) / self.frame_step) + 1
        required = (n_frames - 1) * self.frame_step + self.frame_length
        end_pad = required - total
        padded = torch.nn.functional.pad(waveform, (front_pad, end_pad))
        return padded, n_frames

    def stft(self, waveform: Tensor) -> Tensor:
        """Compute complex STFT.

        Args:
            waveform: (n_channels, n_samples) float tensor on any device.

        Returns:
            Complex tensor of shape (n_channels, n_frames, n_freq=frame_length/2+1).
        """
        if waveform.dim() != 2:
            raise ValueError(f"Expected (n_channels, n_samples), got shape {tuple(waveform.shape)}")
        waveform = waveform.to(self.device)
        padded, _ = self._pad_for_stft(waveform)
        stft_out = torch.stft(
            padded,
            n_fft=self.frame_length,
            hop_length=self.frame_step,
            win_length=self.frame_length,
            window=self._window,
            center=False,
            return_complex=True,
            normalized=False,
        )
        # torch.stft yields (n_channels, freq, frames); transpose to (channels, frames, freq).
        return stft_out.transpose(-1, -2).contiguous()

    def istft(self, stft_t: Tensor, time_crop: int) -> Tensor:
        """Inverse STFT matching tf.signal.inverse_stft + Spleeter's 2/3 factor.

        Implemented manually via per-frame irfft + windowed overlap-add (F.fold) so
        we don't trip torch.istft's NOLA check — Hann periodic has window[0]=0, which
        makes the leading reconstruction position have a zero envelope. TF doesn't
        check; it relies on the caller trimming the leading frame_length samples.

        Args:
            stft_t: Complex tensor (n_channels, n_frames, n_freq).
            time_crop: Number of samples in the original (pre-pad) waveform.

        Returns:
            Real tensor (n_channels, time_crop).
        """
        # Ensure complex64 / float32 (bf16 not supported in irfft).
        stft_t = stft_t.to(torch.complex64)
        # (channels, frames, freq) -> (channels, freq, frames)
        stft_t = stft_t.transpose(-1, -2).contiguous()
        n_channels, n_freq, n_frames = stft_t.shape

        # irfft each frame -> (channels, frame_length, n_frames)
        frames = torch.fft.irfft(stft_t, n=self.frame_length, dim=-2)
        # Apply synthesis window (== analysis window in Spleeter).
        win = self._window.to(torch.float32).view(1, self.frame_length, 1)
        frames = frames * win

        # Overlap-add via F.fold. fold expects (batch, C*kH*kW, n_blocks) and produces
        # (batch, C, H_out, W_out). We treat the time axis as kH with kW=1.
        total_length = (n_frames - 1) * self.frame_step + self.frame_length
        output = torch.nn.functional.fold(
            frames,
            output_size=(total_length, 1),
            kernel_size=(self.frame_length, 1),
            stride=(self.frame_step, 1),
        ).view(n_channels, total_length)
        output = output * WINDOW_COMPENSATION_FACTOR
        return output[..., self.frame_length : self.frame_length + time_crop]

    @staticmethod
    def magnitude(stft_t: Tensor) -> Tensor:
        return stft_t.abs()

    def pad_and_partition(self, spec: Tensor) -> Tensor:
        """Mirror of spleeter.utils.tensor.pad_and_partition.

        Pads the time axis so it's divisible by T, then reshapes
        (channels, frames, freq) -> (n_segments, channels, T, freq).
        """
        if spec.dim() != 3:
            raise ValueError(f"Expected (channels, frames, freq), got {tuple(spec.shape)}")
        channels, frames, freq = spec.shape
        pad_amount = (-frames) % self.T
        if pad_amount:
            spec = torch.nn.functional.pad(spec, (0, 0, 0, pad_amount))
        new_frames = frames + pad_amount
        n_segments = new_frames // self.T
        # (channels, n_segments * T, freq) -> (channels, n_segments, T, freq) -> (n_segments, channels, T, freq)
        spec = spec.view(channels, n_segments, self.T, freq)
        return spec.permute(1, 0, 2, 3).contiguous()

    def crop_freq(self, spec: Tensor) -> Tensor:
        """Crop the frequency axis to F bins (the model's input width)."""
        return spec[..., : self.F]

    def extend_mask_zeros(self, mask: Tensor) -> Tensor:
        """Zero-pad mask from F bins back to frame_length/2+1 bins.

        Mirrors EstimatorSpecBuilder._extend_mask with extension='zeros'.
        """
        n_extra = self.n_freq - self.F
        if n_extra <= 0:
            return mask
        pad = torch.zeros(*mask.shape[:-1], n_extra, dtype=mask.dtype, device=mask.device)
        return torch.cat([mask, pad], dim=-1)

    def reassemble_segments(self, segments: Tensor, n_original_frames: int) -> Tensor:
        """Inverse of pad_and_partition.

        Args:
            segments: (n_segments, channels, T, freq).
            n_original_frames: Number of frames before pad_and_partition.

        Returns:
            (channels, n_original_frames, freq).
        """
        if segments.dim() != 4:
            raise ValueError(f"Expected (n_segments, channels, T, freq), got {tuple(segments.shape)}")
        n_segments, channels, T, freq = segments.shape
        # (n_segments, channels, T, freq) -> (channels, n_segments * T, freq)
        merged = segments.permute(1, 0, 2, 3).reshape(channels, n_segments * T, freq)
        return merged[:, :n_original_frames, :]
