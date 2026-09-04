"""
train_bc_policy.py

Grounding stage, step 3: train one small behavioral-cloning (imitation)
policy per discovered code, using the (state, keypress) pairs bucketed by
label_and_bucket.py.

IMPORTANT: labels here are raw NLD-AA keypress bytes (what the player/bot
actually typed), NOT NLE gym action indices. Converting a predicted
keypress back into an NLE env.step() action index is a separate,
not-yet-verified step -- see the note in the script output. Do not wire
these policies directly into train_option_ppo.py without doing that
conversion and verifying it against your actual installed NLE version.

Reuses FrameEncoder from latent_action_model.py for architectural
consistency with the rest of the pipeline.
"""

import argparse
import os
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state

from latent_action_model import FrameEncoder, NUM_CODES


class BCPolicy(nn.Module):
    """Small CNN classifier: state -> distribution over this code's local
    action vocabulary (built dynamically from the bucket's unique keypresses)."""
    num_classes: int

    @nn.compact
    def __call__(self, state):
        x = FrameEncoder()(state)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.num_classes)(x)
        return logits


def load_bucket(path, max_samples, seed):
    data = np.load(path)
    states = data["states"]
    actions = data["actions"]
    n = states.shape[0]

    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        states = states[idx]
        actions = actions[idx]

    unique_actions = np.unique(actions)
    action_to_idx = {a: i for i, a in enumerate(unique_actions)}
    labels = np.array([action_to_idx[a] for a in actions], dtype=np.int32)

    return states, labels, unique_actions


def train_one_code(code, bucket_path, args):
    print(f"\n--- Code {code} ---")
    states, labels, unique_actions = load_bucket(bucket_path, args.max_samples_per_code, args.seed)
    n = states.shape[0]
    num_classes = len(unique_actions)
    print(f"  {n} samples, {num_classes} distinct keypress values: {unique_actions}")

    counts = np.bincount(labels, minlength=num_classes)
    majority_acc = counts.max() / n
    print(f"  majority-class baseline accuracy: {majority_acc:.3f}")

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    val_size = max(1, int(0.1 * n))
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    train_states, train_labels = states[train_idx], labels[train_idx]
    val_states, val_labels = states[val_idx], labels[val_idx]

    model = BCPolicy(num_classes=num_classes)
    key = jax.random.PRNGKey(args.seed)
    dummy = jnp.zeros((2,) + states.shape[1:], dtype=jnp.int32)  # shape derived from real data, not hardcoded
    params = model.init(key, dummy)["params"]
    tx = optax.adamw(args.lr, weight_decay=args.weight_decay)
    state = train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch_states, batch_labels):
        def loss_fn(params):
            logits = state.apply_fn({"params": params}, batch_states)
            loss = jnp.mean(
                optax.softmax_cross_entropy_with_integer_labels(logits, batch_labels)
            )
            acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
            return loss, acc

        (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss, acc

    @jax.jit
    def eval_batch(state, batch_states, batch_labels):
        logits = state.apply_fn({"params": state.params}, batch_states)
        acc = jnp.mean(jnp.argmax(logits, axis=-1) == batch_labels)
        return acc

    n_train = train_states.shape[0]
    steps_per_epoch = max(1, n_train // args.batch_size)

    # BUGFIX: previously always returned/saved the FINAL epoch's params
    # unconditionally, even when validation accuracy peaked earlier and then
    # declined (overfitting) -- confirmed from bc_train_v2.log: 5 of 8 codes'
    # final-epoch val_acc was LOWER than their epoch-1 val_acc. Now tracks
    # and returns the best-validation-accuracy checkpoint seen during
    # training instead of whatever the last epoch happened to produce.
    best_val_acc = -1.0
    best_params = None
    best_epoch = -1

    for epoch in range(args.epochs):
        epoch_perm = rng.permutation(n_train)
        running_loss, running_acc = 0.0, 0.0
        for i in range(steps_per_epoch):
            idx = epoch_perm[i * args.batch_size:(i + 1) * args.batch_size]
            if len(idx) == 0:
                continue
            bs = jnp.asarray(train_states[idx], dtype=jnp.int32)
            bl = jnp.asarray(train_labels[idx], dtype=jnp.int32)
            state, loss, acc = train_step(state, bs, bl)
            running_loss += float(loss)
            running_acc += float(acc)

        val_bs = jnp.asarray(val_states, dtype=jnp.int32)
        val_bl = jnp.asarray(val_labels, dtype=jnp.int32)
        val_acc = float(eval_batch(state, val_bs, val_bl))

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            best_params = jax.device_get(state.params)
            best_epoch = epoch + 1

        print(f"  epoch {epoch+1}/{args.epochs} | train loss {running_loss/steps_per_epoch:.4f} "
              f"| train acc {running_acc/steps_per_epoch:.3f} | val acc {val_acc:.3f}"
              f"{'  <- new best' if is_best else ''}")

    print(f"  best checkpoint: epoch {best_epoch}, val_acc={best_val_acc:.3f} "
          f"(final epoch was {args.epochs}, val_acc={val_acc:.3f})")

    return {
        "params": best_params,
        "unique_actions": unique_actions,
        "num_classes": num_classes,
        "val_acc": best_val_acc,
        "best_epoch": best_epoch,
        "final_epoch_val_acc": val_acc,
        "majority_baseline_acc": majority_acc,
        "n_samples": n,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket-dir", type=str, default="grounding_buckets")
    parser.add_argument("--out-dir", type=str, default="bc_policies")
    parser.add_argument("--max-samples-per-code", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=1e-3,
        help="AdamW weight decay, regularization variant tested after "
             "observing final-epoch checkpoints overfitting past their peak val_acc.")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    results = {}
    for code in range(NUM_CODES):
        bucket_path = os.path.join(args.bucket_dir, f"bucket_code_{code}.npz")
        if not os.path.exists(bucket_path):
            print(f"\n--- Code {code}: no bucket file found, skipping ---")
            continue

        result = train_one_code(code, bucket_path, args)
        results[code] = result

        out_path = os.path.join(args.out_dir, f"bc_policy_code_{code}.pkl")
        with open(out_path, "wb") as f:
            pickle.dump(result, f)
        print(f"  saved -> {out_path}")

    print("\n=== Summary ===")
    for code, r in results.items():
        lift = r["val_acc"] - r["majority_baseline_acc"]
        print(f"code {code}: n={r['n_samples']:>7} | classes={r['num_classes']:>3} | "
              f"val_acc={r['val_acc']:.3f} | majority_baseline={r['majority_baseline_acc']:.3f} | "
              f"lift={lift:+.3f}")

    print("\nNOTE: labels above are raw keypress bytes, not NLE action indices.")
    print("A verified keypress->action-index mapping is still needed before")
    print("these policies can be wired into train_option_ppo.py.")


if __name__ == "__main__":
    main()
