"""High-level (option-selecting) Gym-style wrapper around NetHackStaircase-v0.

Wraps the primitive NLE environment so that a high-level RL policy selects
among a small discrete set of hard-coded options (see
nethack_option_executor.py) rather than primitive keystrokes. Each call to
step() runs the selected option to completion (many primitive NLE steps
internally) and returns a single high-level transition, with the option's
accumulated primitive reward passed through unchanged (no reward shaping),
so results are directly comparable to a primitive-action baseline.

This wrapper is intentionally NOT JAX-jittable/vmappable: NLE is a real,
stateful Python/C++ game engine reached via gymnasium, so training against
it uses an ordinary synchronous Python loop (see train_option_ppo.py), not
jax.lax.scan over a vectorised environment.

ASSUMPTIONS TO VERIFY LOCALLY (see smoke_test_option_env.py):
- blstats indices below follow NLE's standard NLE_BL_* layout (X, Y, ...,
  HP=10, HPMAX=11, DEPTH=12, GOLD=13, ..., TIME=20, HUNGER=21, ...). This
  is the widely-documented NLE layout, but has not been re-verified
  against your specific installed NLE 1.3.0 build in this session --
  print blstats.shape and a few episodes' values before trusting the
  feature vector for real training.
"""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import nle.env.tasks  # noqa: F401  (registers NLE environments)
import numpy as np
from nle import nethack

from nethack_option_executor import (
    DungeonMemory,
    option_explore,
    option_go_to_downstairs,
    option_descend,
    player_position,
    _stairs_reachable,
    _step,
    WAIT,
)

ENV_ID = "NetHackStaircase-v0"

# Discrete high-level action space.
OPTION_EXPLORE = 0
OPTION_GO_TO_DOWNSTAIRS = 1
OPTION_DESCEND = 2
NUM_OPTIONS = 3

# Observation: a square crop of glyphs centred on the player, plus a small
# numeric feature vector.
CROP_RADIUS = 8  # -> (2*8+1) = 17x17 crop
CROP_SIZE = 2 * CROP_RADIUS + 1

# NLE exposes the true glyph count as nethack.MAX_GLYPH; use one index
# beyond the valid range as an unambiguous "off map" pad value rather than
# guessing a hardcoded number.
MAX_GLYPH = int(nethack.MAX_GLYPH)
PAD_GLYPH = MAX_GLYPH
NUM_GLYPHS = MAX_GLYPH + 1

NUM_FEATURES = 9

# Standard NLE blstats indices (NLE_BL_* constants). See module docstring
# for the "verify locally" caveat.
_BL_HP = 10
_BL_HPMAX = 11
_BL_DEPTH = 12
_BL_GOLD = 13
_BL_TIME = 20
_BL_HUNGER = 21


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


@dataclass
class OptionEnvConfig:
    max_primitive_steps: int = 6000
    """Total primitive NLE steps allowed per high-level episode before
    truncation. This -- not a count of high-level option calls -- is the
    real credit-assignment horizon this project is studying."""
    explore_step_budget: int = 150
    navigation_step_budget: int = 500
    search_repetitions: int = 5
    search_exhaustion_limit: int = 20
    train_seeds: Optional[range] = None
    """If set, each reset() samples a seed from this range for
    deterministic-but-varied training dungeons. If None, NLE seeds itself
    normally (fresh random dungeon every episode)."""


class OptionEnv:
    """Gym-like (reset/step) wrapper exposing 3 discrete options as actions."""

    def __init__(self, config: OptionEnvConfig = OptionEnvConfig()):
        self.config = config
        self.env = gym.make(ENV_ID)
        self.memory: Optional[DungeonMemory] = None
        self._obs = None
        self._primitive_steps_used = 0
        self._rng = np.random.default_rng()

    @property
    def num_options(self) -> int:
        return NUM_OPTIONS

    def _make_observation(self):
        return {
            "crop": _extract_crop(self._obs),
            "features": _extract_features(self._obs, self.memory),
        }

    def reset(self, seed: Optional[int] = None):
        if seed is None and self.config.train_seeds is not None:
            seed = int(self._rng.choice(np.asarray(self.config.train_seeds)))

        if seed is not None:
            # IMPORTANT: env.reset(seed=seed) alone is NOT sufficient for
            # deterministic NLE dungeon generation -- see the debugging
            # notes earlier in this project. NetHack's own RNG must be
            # seeded directly before reset().
            self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)

        obs, info = self.env.reset()
        self._obs = obs
        self.memory = DungeonMemory()
        self.memory.update(obs)
        self._primitive_steps_used = 0

        return self._make_observation(), info

    def _force_wait_turn(self):
        """Consume exactly one real primitive WAIT action.

        Prevents a degenerate exploit: if the chosen option is a no-op
        (e.g. go_to_downstairs before the stairs are known, descend
        while not standing on them, or explore once everything reachable
        has already been exhausted), option_*() correctly returns
        steps=0, reward=0.0 by design. But a zero-cost, zero-progress
        high-level action is a standing incentive for PPO to spam it
        forever -- it never advances primitive_steps_used, so the
        episode can never reach max_primitive_steps and truncate, and
        real exploration/navigation always costs some negative reward
        by comparison (NetHack's per-turn penalty). Observed empirically:
        a first training run completed ZERO episodes across 50,000
        high-level steps because of this loop.

        Charging one real WAIT turn (whatever reward NLE actually
        assigns to it -- not a synthetic penalty) whenever an option
        would otherwise be free matches how a wasted turn should be
        accounted for in a real game, and guarantees every episode's
        primitive-step budget always advances regardless of policy
        behaviour.
        """
        new_obs, reward, done, info = _step(self.env, WAIT)
        if not done:
            self.memory.update(new_obs)
        return new_obs, reward, 1, done

    def step(self, action: int):
        assert self.memory is not None, "call reset() before step()"

        remaining_budget = max(
            self.config.max_primitive_steps - self._primitive_steps_used, 0
        )

        if remaining_budget <= 0:
            return (
                self._make_observation(),
                0.0,
                False,
                True,
                {"option": action, "option_steps": 0, "option_success": False},
            )

        if action == OPTION_EXPLORE:
            budget = min(self.config.explore_step_budget, remaining_budget)
            obs, reward, steps, success, done = option_explore(
                self.env,
                self._obs,
                self.memory,
                max_steps=budget,
                search_repetitions=self.config.search_repetitions,
                search_exhaustion_limit=self.config.search_exhaustion_limit,
            )
        elif action == OPTION_GO_TO_DOWNSTAIRS:
            budget = min(self.config.navigation_step_budget, remaining_budget)
            obs, reward, steps, success, done = option_go_to_downstairs(
                self.env, self._obs, self.memory, max_steps=budget,
            )
        elif action == OPTION_DESCEND:
            # option_descend is a single primitive action; it safely
            # no-ops (0 steps, 0 reward) if the player isn't standing on
            # the downstairs tile, so it never needs a step budget.
            obs, reward, steps, success, done = option_descend(
                self.env, self._obs, self.memory,
            )
        else:
            raise ValueError(f"Unknown option action: {action}")

        if steps == 0 and not done:
            # The chosen option was a complete no-op. Charge one real
            # WAIT turn instead of letting this be a free, zero-progress
            # high-level action (see _force_wait_turn docstring).
            obs, extra_reward, extra_steps, extra_done = self._force_wait_turn()
            reward += extra_reward
            steps += extra_steps
            done = extra_done

        self._obs = obs
        self._primitive_steps_used += steps

        terminated = bool(done)
        truncated = (
            not terminated
            and self._primitive_steps_used >= self.config.max_primitive_steps
        )

        info = {
            "option": action,
            "option_steps": steps,
            "option_success": success,
            "primitive_steps_used": self._primitive_steps_used,
        }

        return self._make_observation(), float(reward), terminated, truncated, info

    def close(self):
        self.env.close()
