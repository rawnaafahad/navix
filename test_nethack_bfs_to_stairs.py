from collections import deque

import gymnasium as gym
import numpy as np
import nle


# ============================================================
# Configuration
# ============================================================

ENV_ID = "NetHackStaircase-v0"

NUM_SEEDS = 20

MAX_DISCOVERY_STEPS = 1000

MAX_BFS_STEPS = 300


# ============================================================
# NLE movement action IDs
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
# Action / coordinate mapping
#
# Coordinates:
#     x = column
#     y = row
# ============================================================

DELTA_TO_ACTION = {
    (0, -1): NORTH,
    (1, 0): EAST,
    (0, 1): SOUTH,
    (-1, 0): WEST,

    (1, -1): NORTHEAST,
    (1, 1): SOUTHEAST,
    (-1, 1): SOUTHWEST,
    (-1, -1): NORTHWEST,
}


NEIGHBOUR_DELTAS = list(
    DELTA_TO_ACTION.keys()
)


# ============================================================
# Observation helpers
# ============================================================

def player_position(obs):

    blstats = np.asarray(
        obs["blstats"]
    )

    return (
        int(blstats[0]),
        int(blstats[1]),
    )


def find_downstairs(obs):

    chars = np.asarray(
        obs["chars"]
    )

    stair_cells = np.argwhere(
        chars == ord(">")
    )

    if len(stair_cells) == 0:

        return None

    row, col = stair_cells[0]

    return (
        int(col),
        int(row),
    )


# ============================================================
# Walkability
# ============================================================

def is_walkable_character(char_value):
    """
    Conservative first-pass walkability test.

    Includes:
        .   floor
        #   corridor
        <   upstairs
        >   downstairs
        +   door
        @   player

    Also allow many object/monster squares because in NetHack
    the visible symbol can temporarily replace the underlying
    floor character.
    """

    char_value = int(
        char_value
    )

    char = chr(
        char_value
    )


    definitely_walkable = {
        ".",
        "#",
        "<",
        ">",
        "+",
        "@",
    }


    if char in definitely_walkable:

        return True


    # --------------------------------------------------------
    # Objects / creatures usually occupy otherwise traversable
    # terrain. We allow common printable symbols and let the
    # environment itself reject genuinely blocked movement.
    # --------------------------------------------------------

    if char.isalpha():

        return True


    if char in {
        "$",
        "%",
        ")",
        "(",
        "[",
        "]",
        "?",
        "!",
        "/",
        "=",
        "*",
        "`",
        ":",
        ";",
        "'",
        '"',
        "&",
    }:

        return True


    return False


def build_walkable_mask(obs):

    chars = np.asarray(
        obs["chars"]
    )

    height, width = (
        chars.shape
    )


    mask = np.zeros(
        (height, width),
        dtype=bool,
    )


    for y in range(
        height
    ):

        for x in range(
            width
        ):

            mask[
                y,
                x,
            ] = is_walkable_character(
                chars[
                    y,
                    x,
                ]
            )


    return mask


# ============================================================
# BFS
# ============================================================

def bfs_path(
    obs,
    start,
    target,
):

    walkable = build_walkable_mask(
        obs
    )

    height, width = (
        walkable.shape
    )


    queue = deque(
        [start]
    )


    parent = {
        start: None
    }


    while queue:

        current = (
            queue.popleft()
        )


        if current == target:

            break


        x, y = current


        for dx, dy in (
            NEIGHBOUR_DELTAS
        ):

            nx = x + dx
            ny = y + dy


            if not (
                0 <= nx < width
                and 0 <= ny < height
            ):

                continue


            next_position = (
                nx,
                ny,
            )


            if next_position in parent:

                continue


            if not walkable[
                ny,
                nx,
            ]:

                continue


            parent[
                next_position
            ] = current


            queue.append(
                next_position
            )


    if target not in parent:

        return None


    path = []

    current = target


    while current is not None:

        path.append(
            current
        )

        current = parent[
            current
        ]


    path.reverse()


    return path


# ============================================================
# Convert path edge to NLE action
# ============================================================

def action_between(
    current,
    next_position,
):

    dx = (
        next_position[0]
        - current[0]
    )

    dy = (
        next_position[1]
        - current[1]
    )


    delta = (
        dx,
        dy,
    )


    if delta not in DELTA_TO_ACTION:

        raise ValueError(
            f"Unsupported movement delta: {delta}"
        )


    return DELTA_TO_ACTION[
        delta
    ]


# ============================================================
# Environment
# ============================================================

env = gym.make(
    ENV_ID
)


rng_global = (
    np.random.default_rng(
        12345
    )
)


successes = 0

stairs_seen_count = 0

bfs_found_count = 0


print(
    "=" * 65
)

print(
    "NETHACK BFS-TO-STAIRS TEST"
)

print(
    "=" * 65
)


# ============================================================
# Test seeds
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


    staircase = None


    # --------------------------------------------------------
    # Temporary random exploration.
    #
    # Only used to expose a staircase so we can test BFS.
    # This will be replaced by EXPLORE later.
    # --------------------------------------------------------

    for discovery_step in range(
        MAX_DISCOVERY_STEPS
    ):

        staircase = find_downstairs(
            obs
        )


        if staircase is not None:

            break


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


    if staircase is None:

        print(
            f"Seed {seed:02d}: "
            f"stairs never discovered"
        )

        continue


    stairs_seen_count += 1


    player = player_position(
        obs
    )


    print()
    print(
        f"Seed {seed:02d}:"
    )

    print(
        "    Player:",
        player
    )

    print(
        "    Stairs:",
        staircase
    )


    # --------------------------------------------------------
    # BFS path
    # --------------------------------------------------------

    path = bfs_path(
        obs,
        player,
        staircase,
    )


    if path is None:

        print(
            "    BFS could not find path"
        )

        continue


    bfs_found_count += 1


    print(
        "    BFS path length:",
        len(path) - 1
    )


    # --------------------------------------------------------
    # Execute BFS path
    # --------------------------------------------------------

    navigation_success = True


    for path_index in range(
        1,
        min(
            len(path),
            MAX_BFS_STEPS + 1,
        ),
    ):

        current = player_position(
            obs
        )

        target_step = path[
            path_index
        ]


        action = action_between(
            current,
            target_step,
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


        new_player = player_position(
            obs
        )


        if new_player != target_step:

            print(
                "    Movement failed:"
            )

            print(
                "        expected:",
                target_step
            )

            print(
                "        actual:",
                new_player
            )

            navigation_success = False

            break


        if (
            terminated
            or truncated
        ):

            navigation_success = False

            break


    # --------------------------------------------------------
    # Descend
    # --------------------------------------------------------

    final_player = player_position(
        obs
    )


    print(
        "    Final player:",
        final_player
    )


    if (
        navigation_success
        and final_player
        == staircase
    ):

        (
            obs,
            reward,
            terminated,
            truncated,
            info,
        ) = env.step(
            DESCEND
        )


        done = (
            terminated
            or truncated
        )


        print(
            "    DESCEND reward:",
            float(
                reward
            )
        )

        print(
            "    Done:",
            done
        )


        if (
            float(reward) > 0.0
            or done
        ):

            successes += 1

            print(
                "    SUCCESS"
            )

        else:

            print(
                "    DESCEND did not finish task"
            )


print()
print(
    "=" * 65
)

print(
    "SUMMARY"
)

print(
    "=" * 65
)


print(
    "Stairs discovered:",
    f"{stairs_seen_count}/{NUM_SEEDS}"
)

print(
    "BFS paths found:",
    f"{bfs_found_count}/{NUM_SEEDS}"
)

print(
    "Successful descents:",
    f"{successes}/{NUM_SEEDS}"
)


env.close()


print()
print(
    "BFS-to-stairs test complete."
)