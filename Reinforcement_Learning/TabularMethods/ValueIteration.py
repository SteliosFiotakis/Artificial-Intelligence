import copy
import random
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from TabularMethods.UsefullClassesAFunctions import GridWorld, visualize

env = GridWorld('HugeMaze.png')
env.starting(1, 1)
env.target(16, 49)
env.print_map()

states_list = env.get_states()


# # # Main Algorithm # # #
gamma = 0.99
theta = 1e-6
policy = {}
value_function = {}
temp_argmax = list()
transition_dict = env.create_transition_matrix()

# Value Function Initialization
for state in states_list:
    if env.terminal_check(state):
        value_function[state] = 0
    else:
        value_function[state] = random.randint(1, 5)

# Value Iteration
while True:
    delta = 0
    for state in states_list:
        if env.terminal_check(state):
            continue
        old_value = value_function[state]
        for action in range(4):
            next_state, reward = transition_dict[state, action]
            temp_argmax.append(reward + gamma * value_function[next_state])
        value_function[state] = max(temp_argmax)
        temp_argmax.clear()
        delta = max(delta, abs(old_value - value_function[state]))
    if delta < theta:
        break

print()
for key, value in value_function.items():
    print(key, value)


for state in states_list:
    for action in range(4):
        next_state, reward = transition_dict[state, action]
        temp_argmax.append(reward + gamma * value_function[next_state])
    policy[state] = temp_argmax.index(max(temp_argmax))
    temp_argmax.clear()

print()
for key, value in policy.items():
    print(key, value)


print()
print('Demonstration:')
print()

def on_click(event):
    if event.button is MouseButton.LEFT:
        if event.inaxes:
            print(event.xdata, event.ydata)
            x_cor = event.xdata
            y_cor = event.ydata
            return x_cor, y_cor


plt.connect('button_press_event', on_click)


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
