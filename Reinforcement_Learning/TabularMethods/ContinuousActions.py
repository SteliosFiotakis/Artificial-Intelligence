import gym
env = gym.make("BipedalWalker-v3", render_mode="human")

print(env.action_space)
print()
print(env.observation_space)

env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    state, reward, terminated, truncated, info = env.step(action)

    print(state)
    print()
    print(action)
    print()
    print()
    print()

    if terminated or truncated:
        env.reset()

env.close()
