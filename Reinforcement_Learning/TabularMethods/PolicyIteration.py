import copy
import random
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from TabularMethods.UsefullClassesAFunctions import GridWorld, visualize

env = GridWorld('HugeMaze.png')
env.starting(1, 1)
env.target(16, 49)
env.print_map()


def make_distribution():
    first = random.uniform(0, 1)
    second = random.uniform(0, 1 - first)
    third = random.uniform(0, 1 - first - second)
    fourth = 1 - first - second - third
    return first, second, third, fourth


def deterministic_action():
    return random.randint(0, 3)


distribution_list = make_distribution()
states_list = env.get_states()


# # # Main Algorithm # # #
gamma = 0.99
theta = 1e-5
policy = {}
value_function = {}

# Policy and Value Function Initialization
for state in states_list:
    if env.terminal_check(state):
        value_function[state] = 0
    else:
        value_function[state] = random.randint(1, 5)
    policy[state] = deterministic_action()

transition_dict = env.create_transition_matrix()
def print_things():
    print('States:')
    for frow in states_list:
        print(frow)
    print()

    print('Transition dictionary:')
    for fkey, fvalue in transition_dict.items():
        print(fkey, fvalue)
    print()

    print('Policy:')
    for fkey, fvalue in policy.items():
        print(fkey, fvalue)
    print()


# print_things()


while True:
    # Policy Evaluation
    while True:
        delta = 0
        for state in states_list:
            if env.terminal_check(state):
                continue
            old_value = value_function[state]
            next_state, reward = transition_dict[state, policy[state]]
            value_function[state] = reward + gamma * value_function[next_state]
            delta = max(delta, abs(old_value - value_function[state]))
        if delta < theta:
            break

    print()
    print('Policy:')
    for key, value in policy.items():
        print(key, value)
    print()

    print('Value Function:')
    for key, value in value_function.items():
        print(key, value)

    # Policy Improvement
    policy_stable = True
    for state in states_list:
        old_action = policy[state]
        argmax_list = list()
        for action in range(4):
            next_state, reward = transition_dict[state, action]
            temp_argmax = reward + gamma * value_function[next_state]
            argmax_list.append(temp_argmax)
        policy[state] = argmax_list.index(max(argmax_list))
        if old_action != policy[state]:
            policy_stable = False
    if policy_stable:
        break

print()
for key, value in policy.items():
    print(key, value)
print()
for key, value in value_function.items():
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
