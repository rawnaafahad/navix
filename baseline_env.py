"""Primitive-action Gym-style wrapper around NetHackStaircase-v0.

This is the CONTROL / BASELINE environment for the options-vs-primitives
comparison this project studies. It exposes NLE's own primitive action
space directly -- no hard-coded options, no BFS pathfinding, no search
logic -- using the exact same observation encoding (cropped glyph window
+ feature vector) as OptionEnv (option_env.py), and is trained with the
exact same PPO loop machinery (train_baseline_ppo.py mirrors
train_option_ppo.py, including the truncation-bootstrapping fix).

Keeping the observation encoding, network architecture, and PPO
implementation identical between this and OptionEnv means any measured
performance/sample-efficiency difference can be attributed to the option
abstraction itself, not to incidental differences elsewhere in the
pipeline.

A DungeonMemory is still maintained here purely so the feature vector
(stairs_known, stairs_reachable, known_frac, exhausted_frac) can be
computed identically to OptionEnv -- this policy never uses it for
planning, only for observing the same kind of information the option
policy sees.
"""

from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import nle.env.tasks  # noqa: F401  (registers NLE environments)
import numpy as np

from nethack_option_executor import DungeonMemory
from option_env import (
    ENV_ID,
    CROP_SIZE,
    NUM_GLYPHS,
    NUM_FEATURES,
    _extract_crop,
    _extract_features,
)


@dataclass
class BaselineEnvConfig:
    max_primitive_steps: int = 6000
    """Same semantics and same default as OptionEnvConfig, for a fair
    comparison: the total number of real NLE turns allowed per episode
    before truncation."""
    train_seeds: Optional[range] = None


class BaselineEnv:
    """Gym-like (reset/step) wrapper exposing NLE's raw action space."""

    def __init__(self, config: BaselineEnvConfig = BaselineEnvConfig()):
        self.config = config
        self.env = gym.make(ENV_ID)
        self.num_actions = int(self.env.action_space.n)
        self.memory: Optional[DungeonMemory] = None
        self._obs = None
        self._primitive_steps_used = 0
        self._rng = np.random.default_rng()

    def _make_observation(self):
        return {
            "crop": _extract_crop(self._obs),
            "features": _extract_features(self._obs, self.memory),
        }

    def reset(self, seed: Optional[int] = None):
        if seed is None and self.config.train_seeds is not None:
            seed = int(self._rng.choice(np.asarray(self.config.train_seeds)))

        if seed is not None:
            # Same deterministic-seeding caveat as OptionEnv: NetHack's
            # own RNG must be seeded directly before reset().
            self.env.unwrapped.seed(core=seed, disp=seed, reseed=False)

        obs, info = self.env.reset()
        self._obs = obs
        self.memory = DungeonMemory()
        self.memory.update(obs)
        self._primitive_steps_used = 0

        return self._make_observation(), info

    def step(self, action: int):
        assert self.memory is not None, "call reset() before step()"

        obs, reward, terminated, truncated_env, info = self.env.step(int(action))
        self._obs = obs
        self._primitive_steps_used += 1

        if not terminated:
            self.memory.update(obs)

        truncated = (
            not terminated
            and (
                truncated_env
                or self._primitive_steps_used >= self.config.max_primitive_steps
            )
        )

        return (
            self._make_observation(),
            float(reward),
            bool(terminated),
            bool(truncated),
            info,
        )

    def close(self):
        self.env.close()