"""PyTorch port of Spleeter's per-instrument U-Net.

Mirrors legacy/spleeter/spleeter/model/functions/unet.py:apply_unet exactly:
- Encoder: 6 strided Conv2D layers with BN + ELU; raw conv outputs are saved for skips.
- Bottom of U: raw conv6 (NOT post-BN) feeds the decoder's first ConvTranspose.
- Decoder: ConvTranspose -> ELU -> BN, with Dropout(0.5) on the first 3 blocks.
- Skips concat with the pre-BN raw conv outputs from the encoder.
- Final: Conv2D(2, 4x4, dilation=2, sigmoid) multiplied with the input spectrogram.

Submodules are registered in the same order Keras's auto-incrementer used, so
state_dict iteration matches TF checkpoint construction order. This is what the
weight converter relies on.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# TF "same" padding for kernel=5, stride=2 is asymmetric: 1 pre, 2 post on each spatial
# axis. PyTorch's `padding=2` would apply symmetric (2, 2) — which produces a
# 1-sample-shifted result that doesn't match the pretrained weights. Verified numerically
# in smoke_test/compare_padding.py (L2/L2 = 1.41 with symmetric, 0.0 with this pad).
_ENC_PAD = (1, 2, 1, 2)  # F.pad order: (W_left, W_right, H_top, H_bottom)
# Conv2DTranspose with TF same crops the full output asymmetrically by [1:-2, 1:-2].
_DEC_CROP_PRE = 1
_DEC_CROP_POST = 2

# Keras defaults (not PyTorch defaults).
_BN_MOMENTUM = 0.01  # PyTorch momentum convention: 1 - keras_momentum (0.99 in Keras).
_BN_EPS = 1e-3


def _activation(name: str) -> nn.Module:
    name = (name or "").lower()
    if name == "elu":
        return nn.ELU()
    if name == "relu":
        return nn.ReLU()
    if name == "leakyrelu":
        return nn.LeakyReLU(0.2)
    # Spleeter default for unspecified is LeakyReLU(0.2).
    return nn.LeakyReLU(0.2)


class UNet(nn.Module):
    """Single-instrument U-Net matching apply_unet(output_mask_logit=False)."""

    def __init__(
        self,
        n_channels: int = 2,
        conv_n_filters: List[int] = [16, 32, 64, 128, 256, 512],
        conv_activation: str = "ELU",
        deconv_activation: str = "ELU",
        dropout: float = 0.5,
        output_logits: bool = False,
    ) -> None:
        super().__init__()
        if len(conv_n_filters) != 6:
            raise ValueError("conv_n_filters must have 6 entries (Spleeter U-Net depth).")
        self.n_channels = n_channels
        self.conv_n_filters = list(conv_n_filters)
        self.dropout_p = dropout
        self.conv_activation_name = conv_activation
        self.deconv_activation_name = deconv_activation
        # When True, forward returns the raw final conv output (no sigmoid, no multiply
        # by input). Used by softmax_unet where masks are computed across stems by an
        # outer softmax, not per-stem sigmoid.
        self.output_logits = output_logits

        # Encoder. Each block is a Conv2D + BatchNorm + ELU; the raw conv output is
        # saved for the skip connection (matches apply_unet using `conv5` not `rel5`).
        # Padding=0 here — we apply asymmetric F.pad(x, _ENC_PAD) in forward.
        in_ch = n_channels
        self.enc_convs = nn.ModuleList()
        self.enc_bns = nn.ModuleList()
        for out_ch in self.conv_n_filters:
            self.enc_convs.append(
                nn.Conv2d(in_ch, out_ch, kernel_size=5, stride=2, padding=0, bias=True)
            )
            self.enc_bns.append(nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM, eps=_BN_EPS))
            in_ch = out_ch
        self.enc_act = _activation(conv_activation)

        # Decoder. ConvTranspose channel pattern is [256, 128, 64, 32, 16, 1].
        # Padding=0, output_padding=0 here — we crop the asymmetric overhang in forward.
        decoder_out_channels = self.conv_n_filters[-2::-1] + [1]
        # The first deconv receives raw conv6 (bottom of U). Subsequent deconvs receive the
        # concatenation [decoder_block_output, encoder_skip], so input channels are doubled.
        self.dec_convs = nn.ModuleList()
        self.dec_bns = nn.ModuleList()
        in_ch = self.conv_n_filters[-1]  # 512 from conv6
        for i, out_ch in enumerate(decoder_out_channels):
            self.dec_convs.append(
                nn.ConvTranspose2d(
                    in_ch,
                    out_ch,
                    kernel_size=5,
                    stride=2,
                    padding=0,
                    output_padding=0,
                    bias=True,
                )
            )
            self.dec_bns.append(nn.BatchNorm2d(out_ch, momentum=_BN_MOMENTUM, eps=_BN_EPS))
            # Next iteration's input: out_ch concatenated with the corresponding encoder skip.
            if i < len(decoder_out_channels) - 1:
                skip_ch = self.conv_n_filters[-2 - i]
                in_ch = out_ch + skip_ch
            else:
                # After the last decoder block we concat with conv1 (n_channels=2 path),
                # but the final Conv2D below operates on `batch12` (the decoder output),
                # NOT on the concat. Matches apply_unet.
                pass
        self.dec_act = _activation(deconv_activation)
        self.dropout = nn.Dropout2d(p=dropout)

        # Final Conv2D: (4,4) kernel, dilation=2, sigmoid. "Same" padding for a 4x4 dilated-2
        # kernel works out to padding=3 (effective kernel = 1 + (4-1)*2 = 7).
        # The input to this final conv is `batch12` (output of the last decoder block) which
        # has 1 channel after the up6 ConvTranspose -> ELU -> BN.
        self.final_conv = nn.Conv2d(
            in_channels=1,
            out_channels=n_channels,
            kernel_size=4,
            stride=1,
            padding=3,
            dilation=2,
            bias=True,
        )
        self.final_act = nn.Sigmoid()

        self._init_weights()

    def _init_weights(self) -> None:
        # Keras he_uniform(seed=50). Match shape via Kaiming uniform.
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_uniform_(m.weight, a=0, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        """Args:
            x: (B, n_channels, T, F) magnitude spectrogram.

        Returns:
            (B, n_channels, T, F) masked spectrogram = sigmoid(conv(decoder_out)) * x.
        """
        # Encoder. Save raw conv outputs for skip connections.
        skips: List[Tensor] = []
        h = x
        for conv, bn in zip(self.enc_convs, self.enc_bns):
            padded = F.pad(h, _ENC_PAD)
            conv_out = conv(padded)
            skips.append(conv_out)
            h = self.enc_act(bn(conv_out))
        # apply_unet feeds the raw conv6 (NOT the post-BN/ELU activation) into the decoder.
        h = skips[-1]

        # Decoder. Apply Dropout on the first 3 blocks. Concat with the encoder skip
        # of the matching resolution (skips[-2], skips[-3], ..., skips[0]).
        n_dec = len(self.dec_convs)
        for i, (deconv, bn) in enumerate(zip(self.dec_convs, self.dec_bns)):
            # TF Conv2DTranspose with same crops the unpadded output [1:-2, 1:-2] on both
            # spatial axes. We replicate by running with padding=0 then slicing manually.
            up = deconv(h)
            up = up[..., _DEC_CROP_PRE:-_DEC_CROP_POST, _DEC_CROP_PRE:-_DEC_CROP_POST]
            up = self.dec_act(up)
            up = bn(up)
            if i < 3:
                up = self.dropout(up)
            if i < n_dec - 1:
                skip = skips[-2 - i]
                h = torch.cat([skip, up], dim=1)
            else:
                # Last decoder block: no skip concat; goes straight to final conv.
                h = up

        out = self.final_conv(h)
        if self.output_logits:
            # softmax_unet path: return raw logits; the outer Stem5UNet softmaxes across stems.
            return out
        return self.final_act(out) * x


class Stem5UNet(nn.Module):
    """Multi-stem container: N independent U-Nets + power-law mask recomputation.

    Two model_type variants matching Spleeter's `apply_unet` / `softmax_unet`:

    * `unet.unet` (default): each U-Net applies `sigmoid(final_conv) * input` itself.
      Used by `spleeter:4stems` and the Carnatic fine-tuning config.
    * `unet.softmax_unet`: each U-Net outputs raw logits; the container stacks across
      stems and applies a softmax across the stem dimension, then multiplies each
      stem's mask by the input. Used by `spleeter:5stems`.

    Regardless of variant, `compute_masks` (used at inference) applies an outer
    power-law renormalisation on the model outputs — matching legacy
    EstimatorSpecBuilder._build_masks. This is independent of the per-stem variant.
    """

    SUPPORTED_MODEL_TYPES = ("unet.unet", "unet.softmax_unet")

    def __init__(
        self,
        instruments: List[str],
        n_channels: int = 2,
        conv_n_filters: List[int] = [16, 32, 64, 128, 256, 512],
        conv_activation: str = "ELU",
        deconv_activation: str = "ELU",
        separation_exponent: int = 2,
        eps: float = 1e-10,
        model_type: str = "unet.unet",
    ) -> None:
        super().__init__()
        if not instruments:
            raise ValueError("instruments must be non-empty")
        if model_type not in self.SUPPORTED_MODEL_TYPES:
            raise ValueError(
                f"Unsupported model_type {model_type!r}. "
                f"Expected one of {self.SUPPORTED_MODEL_TYPES}."
            )
        self._instruments = list(instruments)
        self.separation_exponent = separation_exponent
        self.eps = eps
        self.model_type = model_type
        self._use_softmax_across_stems = model_type == "unet.softmax_unet"
        self.unets = nn.ModuleDict(
            {
                inst: UNet(
                    n_channels=n_channels,
                    conv_n_filters=conv_n_filters,
                    conv_activation=conv_activation,
                    deconv_activation=deconv_activation,
                    output_logits=self._use_softmax_across_stems,
                )
                for inst in self._instruments
            }
        )

    @property
    def instruments(self) -> List[str]:
        return list(self._instruments)

    def forward(self, mix_spec: Tensor) -> Dict[str, Tensor]:
        per_stem = {inst: self.unets[inst](mix_spec) for inst in self._instruments}
        if not self._use_softmax_across_stems:
            return per_stem
        # softmax_unet: per_stem values are raw logits. Stack across a new last dim
        # and softmax across stems, then multiply each mask by the input.
        logits = torch.stack([per_stem[inst] for inst in self._instruments], dim=-1)
        masks = F.softmax(logits.float(), dim=-1).to(logits.dtype)
        return {
            inst: masks[..., i] * mix_spec
            for i, inst in enumerate(self._instruments)
        }

    def compute_masks(self, model_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Power-law mask: (out_i^p + eps/N) / (sum_j out_j^p + eps). Always fp32."""
        p = self.separation_exponent
        outs_f32 = {k: v.float() for k, v in model_outputs.items()}
        powered = {k: v.pow(p) for k, v in outs_f32.items()}
        total = torch.stack(list(powered.values()), dim=0).sum(dim=0) + self.eps
        n = len(self._instruments)
        return {k: (powered[k] + self.eps / n) / total for k in self._instruments}
