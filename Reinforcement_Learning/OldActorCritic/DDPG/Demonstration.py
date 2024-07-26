import gym
import numpy as np
from Agent import Agent


##################################################
# HYPERPARAMETERS
##################################################

env_id = "BipedalWalker-v3"
model_saving_path = f"./Saves/{env_id}/"
learning_plot_saving_path = "Plots/"
replay_buffer_size = 1_000_000
total_steps = 5_000_000
episode_steps = 500
batch_size = 32
gamma = 0.99
load = True
model_n = 33
tau = 5e-03
actor_structure = [512, 512]
critic_structure = [512, 512]
alpha = 1e-03
beta = 2e-03

##################################################
# TRAINING
##################################################

episode = 0
global_step = 0
env = gym.make(env_id, render_mode='human')
agent = Agent(critic_architecture=critic_structure,
              actor_architecture=actor_structure,
              replay_buffer_s=replay_buffer_size,
              model_path=model_saving_path,
              batch_size=batch_size,
              environment=env,
              model_n=model_n,
              do_load=load,
              alpha=alpha,
              gamma=gamma,
              beta=beta,
              tau=tau)

while global_step < total_steps:
    state = env.reset()[0]
    episode_return = 0

    for step in range(1, episode_steps+1):
        global_step += 1

        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)

        state = next_state
        episode_return += reward
        if done:
            break

    print(f"Episode: {episode}, Reward: {episode_return}, Steps: {step}, Global step: {global_step}")
    episode += 1
