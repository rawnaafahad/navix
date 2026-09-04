"""Actor-critic network for the option-level NetHack policy.

Encodes a cropped glyph window (categorical glyph ids) via an embedding +
small CNN, concatenates with a low-dimensional feature vector, and outputs
a Discrete(NUM_OPTIONS) policy plus a scalar value -- reusing flax/distrax,
consistent with the project's existing ActorCritic style. This network is
called on ordinary batched jnp arrays assembled by a plain Python training
loop (see train_option_ppo.py), NOT wrapped in jax.vmap/jax.lax.scan over
a vectorised environment, since the real NLE environment underneath
cannot be jitted or vmapped.
"""

from typing import Tuple

import distrax
import flax.linen as nn
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from jax import Array


class GlyphCropEncoder(nn.Module):
    num_glyphs: int
    embed_dim: int = 16
    hidden_size: int = 128

    @nn.compact
    def __call__(self, crop: Array) -> Array:
        # crop: (batch, H, W) int32 glyph ids
        x = nn.Embed(num_embeddings=self.num_glyphs, features=self.embed_dim)(crop)
        # x: (batch, H, W, embed_dim)
        x = nn.Conv(32, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, kernel_size=(3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return x


class FeatureEncoder(nn.Module):
    hidden_size: int = 32

    @nn.compact
    def __call__(self, features: Array) -> Array:
        x = nn.Dense(self.hidden_size)(features)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.tanh(x)
        return x


class OptionActorCritic(nn.Module):
    num_options: int
    num_glyphs: int
    crop_hidden_size: int = 128
    feature_hidden_size: int = 32
    joint_hidden_size: int = 128

    def setup(self):
        self.crop_encoder = GlyphCropEncoder(
            num_glyphs=self.num_glyphs, hidden_size=self.crop_hidden_size,
        )
        self.feature_encoder = FeatureEncoder(hidden_size=self.feature_hidden_size)

        self.joint = nn.Sequential([nn.Dense(self.joint_hidden_size), nn.relu])

        self.actor_head = nn.Dense(
            self.num_options,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )
        self.critic_head = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0),
        )

    def _encode(self, crop: Array, features: Array) -> Array:
        crop_repr = self.crop_encoder(crop)
        feature_repr = self.feature_encoder(features)
        joint_repr = jnp.concatenate([crop_repr, feature_repr], axis=-1)
        return self.joint(joint_repr)

    def __call__(
        self, crop: Array, features: Array
    ) -> Tuple[distrax.Distribution, Array]:
        h = self._encode(crop, features)
        logits = self.actor_head(h)
        value = jnp.squeeze(self.critic_head(h), axis=-1)
        return distrax.Categorical(logits=logits), value

    def policy(self, crop: Array, features: Array) -> distrax.Distribution:
        h = self._encode(crop, features)
        return distrax.Categorical(logits=self.actor_head(h))

    def value(self, crop: Array, features: Array) -> Array:
        h = self._encode(crop, features)
        return jnp.squeeze(self.critic_head(h), axis=-1)