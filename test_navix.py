import jax
import navix as nx

env = nx.make("Navix-Empty-8x8-v0")

key = jax.random.PRNGKey(0)
timestep = env.reset(key)

print("Environment created successfully")
print("Observation shape:", timestep.observation.shape)
print("Action space:", env.action_space)
print("Initial reward:", timestep.reward)
print("Done:", timestep.is_done())