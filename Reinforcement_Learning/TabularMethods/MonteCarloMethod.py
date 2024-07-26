import random
from TabularMethods.UsefullClassesAFunctions import GridWorld

env = GridWorld('Map3.png')
env.starting(1, 1)
env.target(7, 7)

gamma = 0.99
max_steps = 100

states_list = env.get_states()
policy = {}
value_function = {}
returns = {}

for state in states_list:
    if env.terminal_check(state):
        value_function[state] = 0
    else:
        value_function[state] = random.randint(1, 5)
    policy[state] = random.randint(0, 3)
    returns[state] = list()

cnt = 0
while True:
    cnt += 1
    history = list()
    env.spawn()
    for _ in range(max_steps):
        state = env.observe()
        action = policy[state]
        reward = env.step(action)
        history.append((state, action, reward))
    env.despawn()
    goal = 0
    history.reverse()
    for timestep in history:
        state = timestep[0]
        reward = timestep[2]
        goal = gamma * goal + reward
        if True:
            returns[state].append(goal)
            value_function[state] = sum(returns[state]) / len(returns[state])
    if cnt == 100:
        break


print()
for key, value in policy.items():
    print(key, value)

print()
for key, value in value_function.items():
    print(key, value)
