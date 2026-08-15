from functools import partial
import time
from typing import Callable, Dict, Tuple

import distrax
import jax
import jax.numpy as jnp
from jax import Array
import optax
from flax import struct
from flax.linen import FrozenDict as Params
from flax.training.train_state import TrainState

from navix.observations import rgb
from navix.agents.agent import Agent
from navix.agents.ppo import PPOHparams
from navix.environments import Environment
from navix.environments.environment import Timestep
from navix.states import State
from navix.agents.models import ActorCritic

from jax_option_executor import NUM_OPTIONS, execute_option_jax, option_bootstrap_discount


class OptionBuffer(struct.PyTreeNode):
    done: jax.Array
    action: jax.Array
    reward: jax.Array
    duration: jax.Array
    valid: jax.Array
    log_prob: jax.Array
    obs: jax.Array
    info: Dict[str, jax.Array]
    t: jax.Array
    state: State


class OptionTrainingState(TrainState):
    env_state: Timestep
    rng: jax.Array
    frames: jax.Array
    option_decisions: jax.Array
    epoch: jax.Array
    policy: Callable[[Params, Array], distrax.Distribution] = struct.field(pytree_node=False)
    value_fn: Callable[[Params, Array], Array] = struct.field(pytree_node=False)


class OptionPPO(Agent):
    hparams: PPOHparams
    network: ActorCritic = struct.field(pytree_node=False)
    env: Environment

    def collect_experience(self, train_state: OptionTrainingState):
        def _env_step(collection_state, _):
            env_state, rng = collection_state
            rng, sample_rng = jax.random.split(rng)
            pi = train_state.policy(train_state.params, env_state.observation)
            option_id = jnp.asarray(pi.sample(seed=sample_rng))
            log_prob = jnp.asarray(pi.log_prob(option_id))

            def execute_one(timestep, selected_option):
                return execute_option_jax(
                    self.env,
                    timestep,
                    selected_option,
                    gamma=self.env.gamma,
                )

            new_env_state, duration, option_reward, valid = jax.vmap(
                execute_one, in_axes=(0, 0)
            )(env_state, option_id)

            transition = OptionBuffer(
                done=new_env_state.is_done(),
                action=option_id,
                reward=option_reward,
                duration=duration,
                valid=valid,
                log_prob=log_prob,
                obs=env_state.observation,
                info=new_env_state.info,
                t=env_state.t,
                state=env_state.state,
            )
            return (new_env_state, rng), transition

        (env_state, rng), experience = jax.lax.scan(
            _env_step,
            (train_state.env_state, train_state.rng),
            None,
            self.hparams.num_steps,
        )

        primitive_frames = jnp.asarray(experience.duration.sum(), dtype=jnp.int32)
        option_decisions = jnp.asarray(
            self.hparams.num_steps * self.hparams.num_envs,
            dtype=jnp.int32,
        )

        train_state = train_state.replace(
            env_state=env_state,
            rng=rng,
            frames=train_state.frames + primitive_frames,
            option_decisions=train_state.option_decisions + option_decisions,
        )
        return train_state, experience

    def evaluate_experience(self, train_state, experience, last_val):
        values = jnp.asarray(
            jax.vmap(train_state.value_fn, in_axes=(None, 0))(
                train_state.params, experience.obs
            )
        )
        all_values = jnp.concatenate([values, last_val[None]], axis=0)

        bootstrap_discounts = option_bootstrap_discount(
            experience.duration, self.env.gamma
        ) * (1.0 - experience.done)

        deltas = (
            experience.reward
            + bootstrap_discounts * all_values[1:]
            - all_values[:-1]
        )

        trace_discounts = jnp.power(
            jnp.asarray(self.env.gamma * self.hparams.gae_lambda, dtype=jnp.float32),
            experience.duration,
        ) * (1.0 - experience.done)

        def _backward_gae(next_advantage, inputs):
            delta_t, trace_discount_t = inputs
            advantage_t = delta_t + trace_discount_t * next_advantage
            return advantage_t, advantage_t

        _, advantages = jax.lax.scan(
            _backward_gae,
            jnp.zeros_like(last_val, dtype=jnp.float32),
            (deltas, trace_discounts),
            reverse=True,
        )
        targets = advantages + values
        return values, advantages, targets

    def ppo_loss(self, params, transition_batch, gae, targets, values_old):
        pi, value = jax.vmap(self.network.apply, in_axes=(None, 0))(
            params, transition_batch.obs
        )
        log_prob = pi.log_prob(transition_batch.action)

        if self.hparams.clip_value_loss:
            value_loss = jnp.square(value - targets)
            value_clipped = values_old + jnp.clip(
                value - values_old,
                -self.hparams.clip_eps,
                self.hparams.clip_eps,
            )
            value_loss_clipped = 0.5 * jnp.square(value_clipped - targets)
            value_loss = 0.5 * jnp.maximum(value_loss, value_loss_clipped).mean()
        else:
            value_loss = 0.5 * jnp.square(value - targets).mean()

        ratio = jnp.exp(log_prob - transition_batch.log_prob)
        if self.hparams.normalise_advantage:
            gae = (gae - gae.mean()) / (gae.std() + 1e-8)

        loss_actor1 = ratio * gae
        loss_actor2 = jnp.clip(
            ratio,
            1.0 - self.hparams.clip_eps,
            1.0 + self.hparams.clip_eps,
        ) * gae
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.hparams.vf_coef * value_loss
            - self.hparams.ent_coef * entropy
        )

        logratio = log_prob - transition_batch.log_prob
        approx_kl = ((ratio - 1) - logratio).mean()
        clipfrac = jnp.mean(jnp.abs(ratio - 1.0) > self.hparams.clip_eps)

        logs = {
            "loss/total_loss": total_loss,
            "loss/value_loss": value_loss,
            "loss/actor_loss": loss_actor,
            "loss/entropy": entropy,
            "loss/approx_kl": approx_kl,
            "loss/clipfrac": clipfrac,
        }
        return total_loss, logs

    def sgd_step(self, train_state, minibatch):
        traj_batch, advantages, targets, values_old = minibatch
        grad_fn = jax.value_and_grad(self.ppo_loss, has_aux=True)
        (_, logs), grads = grad_fn(
            train_state.params,
            traj_batch,
            advantages,
            targets,
            values_old,
        )
        train_state = train_state.apply_gradients(grads=grads)
        return train_state, logs

    def _empty_update_logs(self, train_state):
        """
        Return correctly typed placeholder logs after the
        primitive-frame budget has been reached.

        jax.lax.cond requires the active and inactive branches
        to return identical shapes and dtypes.
        """

        float_rollout = jnp.zeros(
            (
                self.hparams.num_steps,
                self.hparams.num_envs,
            ),
            dtype=jnp.float32,
        )

        bool_rollout = jnp.zeros(
            (
                self.hparams.num_steps,
                self.hparams.num_envs,
            ),
            dtype=jnp.bool_,
        )

        int_rollout = jnp.zeros(
            (
                self.hparams.num_steps,
                self.hparams.num_envs,
            ),
            dtype=jnp.int32,
        )

        learning_rate = (
            train_state.opt_state[1]
            .hyperparams["learning_rate"]
        )

        logs = {
            "loss/total_loss": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "loss/value_loss": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "loss/actor_loss": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "loss/entropy": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "loss/approx_kl": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "loss/clipfrac": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "done_mask": bool_rollout,
            "returns": float_rollout,
            "lengths": int_rollout,
            "option/valid_rate": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "option/mean_duration": jnp.asarray(
                0.0,
                dtype=jnp.float32,
            ),
            "iter/frames": train_state.frames,
            "iter/option_decisions": train_state.option_decisions,
            "iter/epochs": train_state.epoch,
            "iter/updates": train_state.step,
            "iter/learning_rate": learning_rate,
            "iter/active": jnp.asarray(
                0,
                dtype=jnp.int32,
            ),
        }

        for option_id in range(NUM_OPTIONS):
            logs[
                f"option/frequency_{option_id}"
            ] = jnp.asarray(
                0.0,
                dtype=jnp.float32,
            )

        return logs

    def _active_update(self, train_state):
        minibatch_size = (
            self.hparams.num_envs
            * self.hparams.num_steps
            // self.hparams.num_minibatches
        )

        train_state, experience = self.collect_experience(train_state)

        for _ in range(self.hparams.num_epochs):
            last_val = train_state.value_fn(
                train_state.params, train_state.env_state.observation
            )
            values, advantages, targets = self.evaluate_experience(
                train_state, experience, last_val
            )

            rng, permutation_rng = jax.random.split(train_state.rng)
            train_state = train_state.replace(rng=rng)

            n_samples = minibatch_size * self.hparams.num_minibatches
            assert n_samples == self.hparams.num_steps * self.hparams.num_envs

            permutation = jax.random.permutation(permutation_rng, n_samples)
            samples = (experience, advantages, targets, values)
            samples = jax.tree.map(
                lambda x: x.reshape((n_samples,) + x.shape[2:]), samples
            )
            shuffled_batch = jax.tree.map(
                lambda x: jnp.take(x, permutation, axis=0), samples
            )
            minibatches = jax.tree.map(
                lambda x: jnp.reshape(
                    x,
                    (self.hparams.num_minibatches, -1) + tuple(x.shape[1:]),
                ),
                shuffled_batch,
            )
            train_state, logs = jax.lax.scan(
                self.sgd_step, train_state, minibatches
            )

        train_state = train_state.replace(
            epoch=train_state.epoch + self.hparams.num_epochs
        )
        logs = jax.tree.map(lambda x: jnp.mean(x), logs)
        learning_rate = train_state.opt_state[1].hyperparams["learning_rate"]

        logs["done_mask"] = experience.done
        logs["returns"] = experience.info["return"]
        logs["lengths"] = experience.t
        logs["option/valid_rate"] = experience.valid.mean()
        logs["option/mean_duration"] = experience.duration.mean()

        for option_id in range(NUM_OPTIONS):
            logs[f"option/frequency_{option_id}"] = jnp.mean(
                experience.action == option_id
            )

        logs["iter/frames"] = train_state.frames
        logs["iter/option_decisions"] = train_state.option_decisions
        logs["iter/epochs"] = train_state.epoch
        logs["iter/updates"] = train_state.step
        logs["iter/learning_rate"] = learning_rate
        logs["iter/active"] = jnp.asarray(1)

        if self.hparams.log_render:
            rng = train_state.rng
            b = jax.random.randint(rng, (), 0, self.hparams.num_envs)
            logs["render/human"] = jax.vmap(rgb)(
                jax.tree.map(lambda x: x[:, b], experience.state)
            ).transpose((0, 3, 1, 2))

        if self.hparams.debug:
            jax.debug.callback(self.log, logs, experience)

        return train_state, logs

    def update(self, train_state, _):
        return jax.lax.cond(
            train_state.frames < self.hparams.budget,
            lambda state: self._active_update(state),
            lambda state: (state, self._empty_update_logs(state)),
            train_state,
        )

    def train(self, rng):
        if self.network.action_dim != NUM_OPTIONS:
            raise ValueError(
                f"OptionPPO requires ActorCritic(action_dim={NUM_OPTIONS}), "
                f"but received action_dim={self.network.action_dim}."
            )

        rng, init_rng = jax.random.split(rng)
        init_x = self.env.observation_space.sample(init_rng)
        params = self.network.init(init_rng, init_x)

        frames_per_min_update = self.hparams.num_steps * self.hparams.num_envs
        num_updates = (
            self.hparams.budget + frames_per_min_update - 1
        ) // frames_per_min_update

        def linear_schedule(count):
            frac = 1.0 - (
                count
                // (self.hparams.num_minibatches * self.hparams.num_epochs)
            ) / num_updates
            return self.hparams.lr * frac

        lr = linear_schedule if self.hparams.anneal_lr else self.hparams.lr
        tx = optax.chain(
            optax.clip_by_global_norm(self.hparams.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(learning_rate=lr, eps=1e-5),
        )

        rng, reset_rng = jax.random.split(rng)
        reset_rngs = jax.random.split(reset_rng, self.hparams.num_envs)
        env_state = jax.vmap(self.env.reset)(reset_rngs)

        train_state = OptionTrainingState.create(
            apply_fn=jax.vmap(self.network.apply, in_axes=(None, 0)),
            params=params,
            tx=tx,
            env_state=env_state,
            rng=rng,
            frames=jnp.asarray(0, dtype=jnp.int32),
            option_decisions=jnp.asarray(0, dtype=jnp.int32),
            epoch=jnp.asarray(0, dtype=jnp.int32),
            policy=jax.vmap(
                partial(self.network.apply, method="policy"),
                in_axes=(None, 0),
            ),
            value_fn=jax.vmap(
                partial(self.network.apply, method="value"),
                in_axes=(None, 0),
            ),
        )

        start_time = time.time()
        train_state, logs = jax.lax.scan(
            self.update,
            train_state,
            None,
            length=num_updates,
        )
        elapsed = time.time() - start_time
        logs["iter/fps"] = jnp.asarray(
            [train_state.frames / max(elapsed, 1e-8)] * num_updates
        )
        return train_state, logs