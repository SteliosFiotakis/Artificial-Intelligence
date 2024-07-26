import copy
import random
from TabularMethods.UsefullClassesAFunctions import GridWorld, visualize


env = GridWorld('AztecMaze.png')
env.starting(1, 1)
env.target(13, 13)

# Parameters
alpha = 1
gamma = 1
epsilon = 1
epsilon_decay = 1e-5
episodes = 100_000
max_steps = 100
e_greedy = list()
temp_max = list()

states_list = env.get_states()
q_function = {}

for state in states_list:
    if env.terminal_check(state):
        q_function[state] = [0, 0, 0, 0]
        continue
    q_function[state] = [random.randint(1, 5),
                         random.randint(1, 5),
                         random.randint(1, 5),
                         random.randint(1, 5)]

for episode in range(episodes):
    env.spawn()
    state = env.observe()
    if random.random() > epsilon:
        action = q_function[state].index(max(q_function[state]))
    else:
        action = random.randint(0, 3)
    for _ in range(max_steps):
        reward = env.step(action)
        next_state = env.observe()
        if random.random() > epsilon:
            next_action = q_function[next_state].index(max(q_function[next_state]))
        else:
            next_action = random.randint(0, 3)
        q_function[state][action] = q_function[state][action] + alpha * \
            (reward + gamma * q_function[next_state][next_action] - q_function[state][action])
        state = next_state
        action = next_action
        if env.terminal_check(state):
            break
    epsilon = epsilon - epsilon_decay


print()
for key, value in q_function.items():
    print(key, value)

print()
print('Demonstration:')
print()


env.starting(1, 1)
map_instances = list()

demonstration_map = copy.deepcopy(env.cmap)
demonstration_map[env.starting_pos[0]][env.starting_pos[1]] = env.starting_id
demonstration_map[env.target_pos[0]][env.target_pos[1]] = env.target_id

map_instances.append(copy.deepcopy(demonstration_map))

env.spawn()

while True:
    print(env.agent_pos)
    state = env.observe()
    demonstration_map[state[0]][state[1]] = env.agent_id
    map_instances.append(copy.deepcopy(demonstration_map))
    if state == env.starting_pos:
        demonstration_map[state[0]][state[1]] = env.starting_id
    else:
        demonstration_map[state[0]][state[1]] = env.floor_id
    if env.terminal_check(state):
        demonstration_map[state[0]][state[1]] = env.target_id
        map_instances.append(copy.deepcopy(demonstration_map))
        break
    action = q_function[state].index(max(q_function[state]))
    env.step(action)
    print(action)
for instance in map_instances:
    visualize(instance)
