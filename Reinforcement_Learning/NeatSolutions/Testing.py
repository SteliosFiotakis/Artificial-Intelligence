import os
import pickle
import neat
import gym
import numpy as np

# load the winner
with open('winner', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)


env = gym.make("Walker2d-v4", render_mode="human")
observation = env.reset()[0]

done = False
while True:
    action = net.activate(observation)

    observation, reward, terminated, truncated, info = env.step(action)
    done = max(terminated, truncated)
