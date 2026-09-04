"""Gym-style wrapper exposing 8 DISCOVERED (grounded) options as a
Discrete(8) action space, matching OptionEnv's interface/observation
format exactly so results are comparable in the 3-way comparison
(primitive vs hand-coded options vs discovered options).

v3 fix: diagnose_discovered_option.py showed pure-greedy (argmax) option
execution gets stuck in a PERMANENT no-op loop -- once the player stops
moving, the input crop stops changing, so argmax deterministically
repeats the exact same action forever (observed: 15 of 16 steps wasted
on one repeated action, explaining a 0% PPO success rate at matched
budget vs. hand-coded options' 15%). Two fixes, both applied here:
  1. Sample from the policy's softmax distribution instead of argmax,
     giving a real chance of escaping a fixed point.
  2. Abort a running option early if the player hasn't moved for
     STUCK_PATIENCE consecutive steps, returning control to the
     high-level PPO policy instead of burning the rest of the budget.

Two DISTINCT representations, as before:
1. OUTER observation (glyphs crop + 9-dim feature vector) -- unchanged
   from v2, matches OptionEnv exactly.
2. INNER representation (tty_chars-based 17x17 crop) -- used only to
   drive the grounded BC policies while an option executes.
"""

import glob
import os
import pickle
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import nle.env.tasks  # noqa: F401
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from nle import nethack

from nethack_option_executor import DungeonMemory, player_position, _stairs_reachable

ENV_ID = "NetHackStaircase-v0"

CROP_RADIUS = 8
CROP_SIZE = 2 * CROP_RADIUS + 1
MAX_GLYPH = int(nethack.MAX_GLYPH)
PAD_GLYPH = MAX_GLYPH
NUM_GLYPHS = MAX_GLYPH + 1
NUM_FEATURES = 9

_BL_HP = 10
_BL_HPMAX = 11
_BL_DEPTH = 12
_BL_GOLD = 13
_BL_TIME = 20
_BL_HUNGER = 21

PLAYER_CHAR = 64  # ord('@')
TTY_CROP_HALF = 8
FUTURE_HORIZON = 16
STUCK_PATIENCE = 3
SAMPLE_TEMPERATURE = 1.0  # tested 1.2/1.5 empirically via compare_temperatures.py -- no meaningful improvement, reverted to no-op


def _extract_features(obs, memory: DungeonMemory) -> np.ndarray:
    blstats = np.asarray(obs["blstats"], dtype=np.float32)

    def _bl(index, default=0.0):
        return float(blstats[index]) if blstats.shape[0] > index else default

    hp = _bl(_BL_HP)
    max_hp = max(_bl(_BL_HPMAX), 1.0)
    depth = _bl(_BL_DEPTH)
    gold = max(_bl(_BL_GOLD), 0.0)
    turns = _bl(_BL_TIME)
    hunger = _bl(_BL_HUNGER)

    stairs_known = 1.0 if memory.downstairs_position() is not None else 0.0
    current = player_position(obs)
    stairs_reachable = 1.0 if _stairs_reachable(memory, current) else 0.0
    known_frac = min(len(memory.known) / 2000.0, 1.0)
    exhausted_frac = min(len(memory.exhausted_frontiers) / 200.0, 1.0)

    return np.array(
        [
            hp / max_hp,
            depth / 30.0,
            np.log1p(gold) / 10.0,
            hunger / 3.0,
            np.tanh(turns / 1000.0),
            stairs_known,
            stairs_reachable,
            known_frac,
            exhausted_frac,
        ],
        dtype=np.float32,
    )


def _extract_crop(obs) -> np.ndarray:
    glyphs = np.asarray(obs["glyphs"])
    height, width = glyphs.shape
    px, py = player_position(obs)

    crop = np.full((CROP_SIZE, CROP_SIZE), PAD_GLYPH, dtype=np.int32)
    for dy in range(-CROP_RADIUS, CROP_RADIUS + 1):
        sy = py + dy
        if not (0 <= sy < height):
            continue
        row_offset = dy + CROP_RADIUS
        for dx in range(-CROP_RADIUS, CROP_RADIUS + 1):
            sx = px + dx
            if not (0 <= sx < width):
                continue
            crop[row_offset, dx + CROP_RADIUS] = int(glyphs[sy, sx])
    return crop


def _find_player_pos_tty(tty_chars):
    positions = np.argwhere(tty_chars == PLAYER_CHAR)
    if len(positions) == 1:
        return int(positions[0][0]), int(positions[0][1])
    return tty_chars.shape[0] // 2, tty_chars.shape[1] // 2


def _crop_tty_centered(grid, row, col, half=TTY_CROP_HALF):
    padded = np.pad(grid, ((half, half), (half, half)), mode="constant", constant_values=0)
    r, c = row + half, col + half
    return padded[r - half:r + half + 1, c - half:c + half + 1]


class FrameEncoder(nn.Module):
    embed_dim: int = 128

    @nn.compact
    def __call__(self, glyph_crop):
        x = nn.Embed(num_embeddings=6000, features=16)(glyph_crop)
        x = nn.Conv(32, (3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(64, (3, 3), strides=(2, 2), padding="SAME")(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(self.embed_dim)(x)
        return nn.relu(x)


class BCPolicy(nn.Module):
    num_classes: int

    @nn.compact
    def __call__(self, state):
        x = FrameEncoder()(state)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_classes)(x)
        return logits


def load_grounded_policies(policy_dir, random_init=False, random_seed=0):
    """Loads per-code BC policies. If random_init=True, keeps the real
    architecture, num_classes, and action vocabulary (unique_actions)
    from the trained bucket files -- but replaces the WEIGHTS with a
    fresh random initialization, isolating grounding quality as the
    sole ablated variable."""
    policies = {}
    paths = sorted(glob.glob(os.path.join(policy_dir, "bc_policy_code_*.pkl")))
    rng_key = jax.random.PRNGKey(random_seed)
    for path in paths:
        code = int(os.path.basename(path).replace("bc_policy_code_", "").replace(".pkl", ""))
        with open(path, "rb") as f:
            r = pickle.load(f)
        model = BCPolicy(num_classes=r["num_classes"])
        action_lookup = jnp.asarray(r["unique_actions"], dtype=jnp.int32)
        if random_init:
            rng_key, init_key = jax.random.split(rng_key)
            dummy = jnp.zeros((1, TTY_CROP_HALF * 2 + 1, TTY_CROP_HALF * 2 + 1), dtype=jnp.int32)
            params = model.init(init_key, dummy)["params"]
        else:
            params = r["params"]

        @jax.jit
        def predict_action_sampled(params, crop, key, model=model, action_lookup=action_lookup):
            logits = model.apply({"params": params}, crop[None, :, :])
            local_idx = jax.random.categorical(key, logits[0] / SAMPLE_TEMPERATURE)
            return action_lookup[local_idx]

        policies[code] = {"params": params, "predict_fn": predict_action_sampled}

    if not policies:
        raise ValueError(f"No bc_policy_code_*.pkl files found in {policy_dir}")
    return policies


@dataclass
class DiscoveredOptionEnvConfig:
    max_primitive_steps: int = 6000
    train_seeds: Optional[range] = None
    policy_dir: str = "bc_policies_cropped"
    use_random_grounding: bool = False
    random_grounding_seed: int = 0
    seed: int = 0
    max_option_steps: Optional[int] = None
    """Diagnostic-only override: caps primitive steps consumed per option
    call, independent of FUTURE_HORIZON and the stuck-loop check. None
    (default) preserves exactly the existing behaviour (cap = FUTURE_HORIZON)
    for every track already reported in this thesis -- this field changes
    NOTHING unless explicitly set. Added to test whether trained options
    returning control less often than random-grounding's observed ~4-6
    primitive-step average per decision explains random-grounding's
    unexpectedly higher held-out success rate (decision-frequency
    hypothesis), as opposed to a genuine grounding-quality effect."""


class DiscoveredOptionEnv:
    def __init__(self, config: DiscoveredOptionEnvConfig = DiscoveredOptionEnvConfig()):
        self.config = config
        self.env = gym.make(ENV_ID)
        self.policies = load_grounded_policies(
            config.policy_dir,
            random_init=config.use_random_grounding,
            random_seed=config.random_grounding_seed,
        )
        self.option_codes = sorted(self.policies.keys())
        self.memory: Optional[DungeonMemory] = None
        self._obs = None
        self._primitive_steps_used = 0
        self._rng = np.random.default_rng()
        self._jax_key = jax.random.PRNGKey(config.seed)

    @property
    def num_options(self) -> int:
        return len(self.option_codes)

    def _make_observation(self):
        return {
            "crop": _extract_crop(self._obs),
            "features": _extract_features(self._obs, self.memory),
        }

    def reset(self, seed: Optional[int] = None):
        if seed is None and self.config.train_seeds is not None:
            seed = int(self._rng.choice(np.asarray(self.config.train_seeds)))

        if seed is not None:
            self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)

        obs, info = self.env.reset()
        self._obs = obs
        self.memory = DungeonMemory()
        self.memory.update(obs)
        self._primitive_steps_used = 0

        return self._make_observation(), info

    def step(self, action: int):
        assert self.memory is not None, "call reset() before step()"

        code = self.option_codes[action]
        policy = self.policies[code]

        remaining_budget = max(
            self.config.max_primitive_steps - self._primitive_steps_used, 0
        )
        if remaining_budget <= 0:
            return (
                self._make_observation(),
                0.0,
                False,
                True,
                {"option": action, "option_code": code, "option_steps": 0},
            )

        option_step_cap = (
            self.config.max_option_steps
            if self.config.max_option_steps is not None
            else FUTURE_HORIZON
        )
        steps_budget = min(option_step_cap, remaining_budget)
        total_reward = 0.0
        steps_taken = 0
        terminated = False
        stuck_count = 0
        prev_pos = None

        for _ in range(steps_budget):
            tty_chars = np.asarray(self._obs["tty_chars"])
            row, col = _find_player_pos_tty(tty_chars)

            if prev_pos is not None and (row, col) == prev_pos:
                stuck_count += 1
            else:
                stuck_count = 0
            prev_pos = (row, col)

            if stuck_count >= STUCK_PATIENCE:
                break

            crop = _crop_tty_centered(tty_chars, row, col).astype(np.int32)
            crop_j = jnp.asarray(crop)

            self._jax_key, sample_key = jax.random.split(self._jax_key)
            action_idx = int(policy["predict_fn"](policy["params"], crop_j, sample_key))

            obs, reward, term, trunc, info = self.env.step(action_idx)
            self._obs = obs
            self.memory.update(obs)
            total_reward += float(reward)
            steps_taken += 1

            if term or trunc:
                terminated = bool(term)
                break

        self._primitive_steps_used += steps_taken

        truncated = (
            not terminated
            and self._primitive_steps_used >= self.config.max_primitive_steps
        )

        info = {
            "option": action,
            "option_code": code,
            "option_steps": steps_taken,
            "primitive_steps_used": self._primitive_steps_used,
            "stuck_abort": stuck_count >= STUCK_PATIENCE,
        }

        return self._make_observation(), total_reward, terminated, truncated, info

    def close(self):
        self.env.close()
