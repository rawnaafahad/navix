"""
latent_action_model.py

Discovers discrete latent "actions" from actionless (state-only) NetHack
transitions. Trained on NLD-AA with the keypresses column hidden.

Architecture: encode past frame -> project+quantize a discrete code from
(past, future) jointly during training -> decode a DOWNSAMPLED future
glyph grid from (past_embed, code) via cross-entropy reconstruction.

Reconstructing raw (downsampled) future observations, rather than matching
a self-referential embedding produced by the same encoder, is deliberate:
embedding-matching lets the encoder cheat by collapsing to a constant
output, trivially driving loss to zero without learning anything (observed
empirically: loss -> 0.0000, codebook collapsed to 1 code even with
dead-code reset firing). Predicting a coarse 6x20 grid (4x downsample from
24x80) keeps the anti-collapse property while being far cheaper than a
full-resolution decoder, which was observed to bottleneck CPU training
(>40 minutes, zero completed logging intervals).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import Tuple

NUM_CODES = 8
CODE_DIM = 32
FUTURE_HORIZON = 16
EMBED_DIM = 128
NUM_GLYPHS = 6000  # NetHack has ~5976 distinct glyphs; padded up for safety
TARGET_H = 6        # downsampled reconstruction target height (24 / 4)
TARGET_W = 20        # downsampled reconstruction target width  (80 / 4)


class FrameEncoder(nn.Module):
    """Small CNN over the glyph grid, reused from OptionActorCritic's shape."""
    embed_dim: int = EMBED_DIM

    @nn.compact
    def __call__(self, glyph_crop):
        x = nn.Embed(num_embeddings=NUM_GLYPHS, features=16)(glyph_crop)
        x = nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.embed_dim)(x)
        return nn.relu(x)


class VectorQuantizer(nn.Module):
    """Standard VQ-VAE codebook layer (van den Oord et al.)."""
    num_codes: int = NUM_CODES
    code_dim: int = CODE_DIM
    commitment_cost: float = 0.25

    @nn.compact
    def __call__(self, z_e):
        codebook = self.param(
            "codebook",
            nn.initializers.uniform(scale=1.0 / self.num_codes),
            (self.num_codes, self.code_dim),
        )
        dists = (
            jnp.sum(z_e ** 2, axis=1, keepdims=True)
            - 2 * jnp.dot(z_e, codebook.T)
            + jnp.sum(codebook ** 2, axis=1)
        )
        code_indices = jnp.argmin(dists, axis=1)
        z_q = codebook[code_indices]

        z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

        codebook_loss = jnp.mean((jax.lax.stop_gradient(z_e) - z_q) ** 2)
        commitment_loss = jnp.mean((z_e - jax.lax.stop_gradient(z_q)) ** 2)
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss

        return z_q_st, code_indices, vq_loss


class FutureDecoder(nn.Module):
    """Reconstructs a DOWNSAMPLED future glyph grid from (past_embed, code).
    Outputs per-cell logits over the glyph vocabulary: (B, TARGET_H, TARGET_W, NUM_GLYPHS).
    Much cheaper than full 24x80 resolution while keeping the same
    anti-collapse property (real, diverse reconstruction target).
    """
    target_h: int = TARGET_H
    target_w: int = TARGET_W
    hidden_dim: int = 128
    num_glyphs: int = NUM_GLYPHS

    @nn.compact
    def __call__(self, past_embed, z_q):
        x = jnp.concatenate([past_embed, z_q], axis=-1)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.Dense(self.target_h * self.target_w * 16)(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], self.target_h, self.target_w, 16))
        logits = nn.Dense(self.num_glyphs)(x)  # (B, target_h, target_w, num_glyphs)
        return logits


def downsample_glyphs(glyph_grid, target_h=TARGET_H, target_w=TARGET_W):
    """Downsample a (B, 24, 80) glyph grid to (B, target_h, target_w) by
    taking a strided subsample (nearest-neighbor style; glyph ids are
    categorical, so averaging/interpolation would be meaningless)."""
    B, H, W = glyph_grid.shape
    row_idx = jnp.linspace(0, H - 1, target_h).astype(jnp.int32)
    col_idx = jnp.linspace(0, W - 1, target_w).astype(jnp.int32)
    return glyph_grid[:, row_idx][:, :, col_idx]


class LatentActionModel(nn.Module):
    num_codes: int = NUM_CODES
    code_dim: int = CODE_DIM

    def setup(self):
        self.encoder = FrameEncoder()
        self.vq = VectorQuantizer(num_codes=self.num_codes, code_dim=self.code_dim)
        self.code_proj = nn.Dense(self.code_dim)
        self.decoder = FutureDecoder()

    def __call__(self, past_frame, future_frame):
        past_embed = self.encoder(past_frame)
        future_embed = self.encoder(future_frame)

        z_e = self.code_proj(jnp.concatenate([past_embed, future_embed], axis=-1))
        z_q, code_indices, vq_loss = self.vq(z_e)

        logits = self.decoder(past_embed, z_q)  # (B, TARGET_H, TARGET_W, num_glyphs)
        target = downsample_glyphs(future_frame)  # (B, TARGET_H, TARGET_W)
        recon_loss = jnp.mean(
            optax.softmax_cross_entropy_with_integer_labels(logits, target)
        )

        total_loss = recon_loss + vq_loss
        return total_loss, code_indices

    def encode_z_e(self, past_frame, future_frame):
        """Expose the pre-quantization vector z_e directly, so the training
        loop can use real encoder outputs to reseed dead codebook entries."""
        past_embed = self.encoder(past_frame)
        future_embed = self.encoder(future_frame)
        z_e = self.code_proj(jnp.concatenate([past_embed, future_embed], axis=-1))
        return z_e


    def assign_code(self, past_frame, future_frame):
        """Cheap code-only inference: encoder + VQ, no decoder. Used for
        labeling large amounts of data where we only need the discrete
        code index, not a reconstruction."""
        past_embed = self.encoder(past_frame)
        future_embed = self.encoder(future_frame)
        z_e = self.code_proj(jnp.concatenate([past_embed, future_embed], axis=-1))
        _, code_indices, _ = self.vq(z_e)
        return code_indices
