import gym
# import random
import numpy as np

env = gym.make("FrozenLake-v1", is_slippery=False)

# policy = dict()
q_table = dict()

for i in range(16):
    # policy[i] = random.randint(0, 3)
    # for number in range(4):
    q_table[i] = list((0, 0, 0, 0))

episodes = 1000
steps = 20
alpha = 0.1
gamma = 1

for episode in range(episodes):
    state = env.reset()[0]
    done = False

    for step in range(steps):
        # action = policy[state]
        action = np.argmax(q_table[state])
        next_state, reward, done, _, info = env.step(action)
        q_table[state][action] = q_table[state][action] + alpha * \
            (reward + gamma * q_table[next_state][action] - q_table[state][action])
        if done:
            break
        state = next_state


for episode in range(episodes):
    state = env.reset()[0]
    done = False

    for step in range(steps):
        action = np.argmax(q_table[state])
        next_state, reward, done, _, info = env.step(action)
        env.render()
        if done:
            break
        state = next_state
