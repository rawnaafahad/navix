import jax
import navix as nx

from option_executor import (
    execute_option,
    OPTION_NAMES,
)


env = nx.make(
    "Navix-DoorKey-Random-8x8-v0"
)

rng = jax.random.PRNGKey(0)
timestep = env.reset(rng)


# Correct option sequence
sequence = [
    0,  # GO_TO_KEY
    1,  # PICKUP_KEY
    2,  # GO_TO_DOOR
    3,  # OPEN_DOOR
    4,  # GO_TO_GOAL
]


total_primitive_steps = 0


print("Testing option executor...\n")


for option_id in sequence:

    print(
        "Executing:",
        OPTION_NAMES[option_id]
    )

    timestep, duration, valid = execute_option(
        env,
        timestep,
        option_id
    )

    total_primitive_steps += duration

    print(
        "Valid:",
        valid
    )

    print(
        "Duration:",
        duration
    )

    print()


print(
    "Reward:",
    timestep.reward
)

print(
    "Done:",
    timestep.is_done()
)

print(
    "Total primitive steps:",
    total_primitive_steps
)


assert bool(
    timestep.is_done()
)

assert float(
    timestep.reward
) > 0.0


print(
    "\nOption executor smoke test passed."
)