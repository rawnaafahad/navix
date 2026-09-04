import sys
import numpy as np
import gymnasium as gym
import nle


# ============================================================
# CONFIGURATION
# ============================================================

ENV_ID = "NetHackStaircase-v0"

ROLLOUT_SEEDS = [0, 1, 2]

MAX_STEPS = 200


# ============================================================
# HEADER
# ============================================================

print("=" * 60)
print("NETHACK / NLE ENVIRONMENT SANITY CHECK")
print("=" * 60)

print("Python:", sys.version)
print("NLE version:", nle.__version__)
print("Gymnasium version:", gym.__version__)

print()
print("Environment:", ENV_ID)


# ============================================================
# CREATE ENVIRONMENT
# ============================================================

print()
print("=" * 60)
print("CREATING ENVIRONMENT")
print("=" * 60)

try:
    env = gym.make(ENV_ID)

except Exception as error:
    print()
    print("FAILED TO CREATE ENVIRONMENT")
    print(type(error).__name__ + ":", error)
    raise


print("Environment created successfully.")


# ============================================================
# ENVIRONMENT SPACES
# ============================================================

print()
print("=" * 60)
print("ENVIRONMENT SPACES")
print("=" * 60)

print()
print("Action space:")
print(env.action_space)

print()
print("Observation space:")
print(env.observation_space)

if hasattr(env.action_space, "n"):
    print()
    print(
        "Number of discrete actions:",
        env.action_space.n,
    )


# ============================================================
# ACTION SET
# ============================================================

print()
print("=" * 60)
print("ACTION SET")
print("=" * 60)

base_env = env.unwrapped

actions = getattr(
    base_env,
    "actions",
    None,
)

if actions is None:
    print("Environment does not expose an actions attribute.")

else:
    print("Number of actions:", len(actions))
    print()

    for action_id, action in enumerate(actions):
        print(
            f"{action_id:>3}: {action}"
        )


# ============================================================
# INITIAL OBSERVATION
# ============================================================

print()
print("=" * 60)
print("INITIAL OBSERVATION")
print("=" * 60)

observation, info = env.reset(seed=0)

print()
print("Observation type:", type(observation))

print()
print("Reset info:")
print(info)


if isinstance(observation, dict):

    print()
    print("Observation keys:")

    for key in observation.keys():
        print(" -", key)

    print()
    print("Observation shapes:")

    for key, value in observation.items():

        array = np.asarray(value)

        print(
            f"{key:>20}: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}"
        )

else:

    array = np.asarray(observation)

    print()
    print(
        "Observation shape:",
        array.shape,
    )

    print(
        "Observation dtype:",
        array.dtype,
    )


# ============================================================
# BLSTATS
# ============================================================

if (
    isinstance(observation, dict)
    and "blstats" in observation
):

    blstats = np.asarray(
        observation["blstats"]
    )

    print()
    print("=" * 60)
    print("BLSTATS")
    print("=" * 60)

    print()
    print("Shape:", blstats.shape)
    print("Values:")
    print(blstats)

    if len(blstats) >= 2:

        print()
        print(
            "Player position:"
        )

        print(
            "x =",
            int(blstats[0]),
        )

        print(
            "y =",
            int(blstats[1]),
        )


# ============================================================
# MESSAGE
# ============================================================

def decode_message(message_array):

    try:

        raw = bytes(
            np.asarray(
                message_array,
                dtype=np.uint8,
            ).tolist()
        )

        return (
            raw.split(
                b"\x00",
                1,
            )[0]
            .decode(
                "utf-8",
                errors="replace",
            )
        )

    except Exception:

        return str(
            message_array
        )


if (
    isinstance(observation, dict)
    and "message" in observation
):

    print()
    print("=" * 60)
    print("INITIAL MESSAGE")
    print("=" * 60)

    print()
    print(
        decode_message(
            observation["message"]
        )
    )


# ============================================================
# GLYPH / MAP INFORMATION
# ============================================================

print()
print("=" * 60)
print("MAP OBSERVATION CHECK")
print("=" * 60)

map_keys = [
    "glyphs",
    "chars",
    "colors",
    "specials",
]

for key in map_keys:

    if (
        isinstance(observation, dict)
        and key in observation
    ):

        array = np.asarray(
            observation[key]
        )

        print(
            f"{key:>10}: "
            f"shape={array.shape}, "
            f"dtype={array.dtype}"
        )


# ============================================================
# RANDOM ROLLOUTS
# ============================================================

print()
print("=" * 60)
print("RANDOM POLICY ROLLOUTS")
print("=" * 60)

episode_returns = []
episode_lengths = []


for seed in ROLLOUT_SEEDS:

    observation, info = env.reset(
        seed=seed
    )

    total_reward = 0.0
    episode_steps = 0
    done = False

    print()
    print("-" * 40)
    print("SEED:", seed)
    print("-" * 40)

    for step_index in range(MAX_STEPS):

        action = env.action_space.sample()

        (
            observation,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(action)

        done = (
            terminated
            or truncated
        )

        total_reward += float(reward)
        episode_steps += 1

        # Print only the first ten transitions.
        if step_index < 10:

            message = ""

            if (
                isinstance(observation, dict)
                and "message" in observation
            ):

                message = decode_message(
                    observation["message"]
                )

            print(
                f"Step {step_index + 1:>3} | "
                f"action={action:>3} | "
                f"reward={float(reward):>7.3f} | "
                f"done={done}"
            )

            if message:
                print(
                    "       message:",
                    message,
                )

        if done:
            break


    episode_returns.append(
        total_reward
    )

    episode_lengths.append(
        episode_steps
    )

    print()
    print(
        f"Episode finished | "
        f"return={total_reward:.3f} | "
        f"steps={episode_steps} | "
        f"done={done}"
    )


# ============================================================
# SUMMARY
# ============================================================

episode_returns = np.asarray(
    episode_returns,
    dtype=np.float32,
)

episode_lengths = np.asarray(
    episode_lengths,
    dtype=np.float32,
)


print()
print("=" * 60)
print("RANDOM ROLLOUT SUMMARY")
print("=" * 60)

print()
print(
    "Episodes:",
    len(episode_returns),
)

print(
    "Returns:",
    episode_returns.tolist(),
)

print(
    "Lengths:",
    episode_lengths.tolist(),
)

print(
    "Mean return:",
    float(
        episode_returns.mean()
    ),
)

print(
    "Mean rollout length:",
    float(
        episode_lengths.mean()
    ),
)


# ============================================================
# FINAL OBSERVATION DIAGNOSTIC
# ============================================================

if (
    isinstance(observation, dict)
    and "blstats" in observation
):

    final_blstats = np.asarray(
        observation["blstats"]
    )

    print()
    print("Final blstats:")
    print(final_blstats)


if (
    isinstance(observation, dict)
    and "message" in observation
):

    print()
    print("Final message:")

    print(
        decode_message(
            observation["message"]
        )
    )


# ============================================================
# CLEANUP
# ============================================================

env.close()


print()
print("=" * 60)
print("NETHACK ENVIRONMENT SANITY CHECK COMPLETE")
print("=" * 60)

print()
print(
    "If this script completes successfully, the next stage is "
    "to inspect the action/observation structure and design "
    "the first hard-coded NetHack options."
)