import jax
import jax.numpy as jnp
import numpy as np

import navix as nx
from navix import observations
from navix.agents import PPOHparams, ActorCritic
from navix.environments.environment import Environment

from option_ppo import OptionPPO


def FlattenObsWrapper(env: Environment):
    flatten_obs_fn = lambda x: jnp.ravel(env.observation_fn(x))
    flatten_obs_shape = (int(np.prod(env.observation_space.shape)),)
    return env.replace(
        observation_fn=flatten_obs_fn,
        observation_space=env.observation_space.replace(shape=flatten_obs_shape),
    )


env_id = "Navix-DoorKey-Random-8x8-v0"
env = nx.make(
    env_id,
    observation_fn=observations.symbolic_first_person,
    gamma=0.99,
)
env = FlattenObsWrapper(env)

config = PPOHparams().replace(
    budget=512,
    num_envs=4,
    num_steps=8,
    num_minibatches=4,
    num_epochs=1,
    anneal_lr=False,
)

agent = OptionPPO(
    hparams=config,
    network=ActorCritic(action_dim=5),
    env=env,
)

print("Environment:", env_id)
print("Observation shape:", env.observation_space.shape)
print("High-level action count:", agent.network.action_dim)
print("Primitive-frame budget:", config.budget)
print("\nStarting Option PPO smoke test...")

rng = jax.random.PRNGKey(0)
train_state, logs = agent.train(rng)

print("\nOption PPO smoke test finished.")
print("Final primitive frames:", train_state.frames)
print("Final option decisions:", train_state.option_decisions)
print("Final optimiser step:", train_state.step)

active = np.asarray(logs["iter/active"]).astype(bool)
frames = np.asarray(logs["iter/frames"])
valid_rate = np.asarray(logs["option/valid_rate"])
mean_duration = np.asarray(logs["option/mean_duration"])

print("Number of active PPO updates:", int(active.sum()))

if active.any():
    print("Last active primitive frame count:", frames[active][-1])
    print("Last active option valid rate:", valid_rate[active][-1])
    print("Last active mean option duration:", mean_duration[active][-1])

assert int(train_state.frames) >= config.budget
assert int(train_state.option_decisions) > 0
assert np.isfinite(np.asarray(logs["loss/total_loss"])[active]).all()

print("\nOPTION PPO SMOKE TEST PASSED")