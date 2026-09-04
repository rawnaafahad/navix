import numpy as np
import gymnasium as gym
import nle


ENV_ID = "NetHackStaircase-v0"

NUM_SEEDS = 20
MAX_STEPS = 500


# ============================================================
# NetHack reduced action IDs
# ============================================================

NORTH = 1
EAST = 2
SOUTH = 3
WEST = 4

NORTHEAST = 5
SOUTHEAST = 6
SOUTHWEST = 7
NORTHWEST = 8

DESCEND = 18
WAIT = 19


MOVEMENT_ACTIONS = [
    NORTH,
    EAST,
    SOUTH,
    WEST,
    NORTHEAST,
    SOUTHEAST,
    SOUTHWEST,
    NORTHWEST,
]


# ============================================================
# Helpers
# ============================================================

def player_position(obs):

    blstats = np.asarray(
        obs["blstats"]
    )

    x = int(
        blstats[0]
    )

    y = int(
        blstats[1]
    )

    return (
        x,
        y,
    )


def find_downstairs(obs):
    """
    Search the visible ASCII map for the NetHack downstairs
    character: '>' (ASCII 62).

    Returns:
        list of (x, y) positions
    """

    chars = np.asarray(
        obs["chars"]
    )

    stair_locations = np.argwhere(
        chars == ord(">")
    )

    positions = []

    for row, col in stair_locations:

        # NLE map array:
        #   row -> y
        #   col -> x

        positions.append(
            (
                int(col),
                int(row),
            )
        )

    return positions


def decode_message(obs):

    raw = bytes(
        np.asarray(
            obs["message"],
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


# ============================================================
# Environment
# ============================================================

env = gym.make(
    ENV_ID
)


print(
    "=" * 60
)

print(
    "NETHACK DOWNSTAIRS DETECTION TEST"
)

print(
    "=" * 60
)

print(
    "Environment:",
    ENV_ID
)

print(
    "Seeds:",
    NUM_SEEDS
)

print()


seeds_where_stairs_seen = 0

successful_descents = 0

first_seen_steps = []


# ============================================================
# Rollouts
# ============================================================

for seed in range(
    NUM_SEEDS
):

    obs, info = env.reset(
        seed=seed
    )

    rng = np.random.default_rng(
        seed
    )

    seen_stairs = False

    first_seen = None

    descended = False


    for step in range(
        MAX_STEPS
    ):

        player = player_position(
            obs
        )

        stairs = find_downstairs(
            obs
        )


        if (
            len(stairs) > 0
            and not seen_stairs
        ):

            seen_stairs = True

            first_seen = step

            print(
                f"Seed {seed:02d}: "
                f"stairs first seen at step {step}"
            )

            print(
                "    Player:",
                player
            )

            print(
                "    Stair locations:",
                stairs
            )


        # ----------------------------------------------------
        # If standing on visible downstairs, try descending.
        # ----------------------------------------------------

        if player in stairs:

            obs, reward, terminated, truncated, info = (
                env.step(
                    DESCEND
                )
            )

            message = decode_message(
                obs
            )

            done = (
                terminated
                or truncated
            )

            print(
                f"    DESCEND action | "
                f"reward={float(reward):.3f} | "
                f"done={done}"
            )

            if message:

                print(
                    "    Message:",
                    message
                )


            if (
                float(reward) > 0
                or done
            ):

                descended = True

                break


        # ----------------------------------------------------
        # Random movement only.
        #
        # Important:
        # no eat/search/menu-producing commands.
        # ----------------------------------------------------

        action = int(
            rng.choice(
                MOVEMENT_ACTIONS
            )
        )


        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            action
        )


        if (
            terminated
            or truncated
        ):

            break


    if seen_stairs:

        seeds_where_stairs_seen += 1

        first_seen_steps.append(
            first_seen
        )


    if descended:

        successful_descents += 1


    print(
        f"Seed {seed:02d} summary: "
        f"stairs_seen={seen_stairs} | "
        f"descended={descended}"
    )


# ============================================================
# Summary
# ============================================================

print()
print(
    "=" * 60
)

print(
    "STAIR DETECTION SUMMARY"
)

print(
    "=" * 60
)


print(
    "Stairs observed:",
    f"{seeds_where_stairs_seen}/{NUM_SEEDS}"
)


print(
    "Successful descents:",
    f"{successful_descents}/{NUM_SEEDS}"
)


if len(
    first_seen_steps
) > 0:

    print(
        "Mean first-seen step:",
        float(
            np.mean(
                first_seen_steps
            )
        )
    )

    print(
        "Min first-seen step:",
        int(
            np.min(
                first_seen_steps
            )
        )
    )

    print(
        "Max first-seen step:",
        int(
            np.max(
                first_seen_steps
            )
        )
    )


env.close()


print()
print(
    "Downstairs detection test complete."
)