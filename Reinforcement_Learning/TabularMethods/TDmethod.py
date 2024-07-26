import random
from TabularMethods.UsefullClassesAFunctions import GridWorld

env = GridWorld('HugeMaze.png')
env.starting(1, 1)
env.target(7, 7)

gamma = 0.99
alpha = 0.1
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

cnt = 0
    returns[state] = list()
while True:



print()
for key, value in policy.items():
    print(key, value)

print()
for key, value in value_function.items():
    print(key, value)
