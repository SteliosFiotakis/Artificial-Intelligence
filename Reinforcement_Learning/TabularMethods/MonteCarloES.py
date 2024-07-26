import copy
import random
from TabularMethods.UsefullClassesAFunctions import GridWorld, visualize


env = GridWorld('SmallMap.png')
env.starting(1, 1)
env.target(5, 5)

gamma = 0.99
max_steps = 100

# Parameters
states_list = env.get_states()
print(len(states_list))
policy = {}
q_function = {}
returns = {}

for state in states_list:
    policy[state] = random.randint(0, 3)
    for action in range(4):
        q_function[(state, action)] = random.randint(1, 5)
        returns[(state, action)] = list()


cnt = 0
while True:
    cnt += 1
    for starting_state in states_list:
        for starting_action in range(4):
            history = list()
            visited = list()
            env.starting(starting_state[0], starting_state[1])
            env.spawn()
            state = env.observe()
            action = starting_action
            reward = env.step(action)
            history.append((state, action, reward))
            visited.append((state, action))
            for _ in range(max_steps):
                state = env.observe()
                action = policy[state]
                reward = env.step(action)
                history.append((state, action, reward))
                visited.append((state, action))
            env.despawn()
            history.reverse()
            visited.reverse()
            goal = 0
            for t_idx, timestep in enumerate(history):
                state = timestep[0]
                action = timestep[1]
                reward = timestep[2]
                goal = gamma * goal + reward
                if (state, action) not in visited[t_idx+1:]:
                    returns[(state, action)].append(goal)
                    q_function[(state, action)] = sum(returns[(state, action)]) / len(returns[(state, action)])
                    temp_argmax = list()
                    for sub_action in range(4):
                        temp_argmax.append(q_function[(state, sub_action)])
                    policy[state] = temp_argmax.index(max(temp_argmax))
                    temp_argmax.clear()
    if cnt == 100:
        break


print()
for key, value in policy.items():
    print(key, value)

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
    env.step(policy[state])

for instance in map_instances:
    visualize(instance)
