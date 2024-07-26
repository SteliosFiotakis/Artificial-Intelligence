import gym
import numpy as np
from Agent import Agent
from utils import plot_learning_curve


##################################################
# HYPERPARAMETERS
##################################################

env_id = "Walker2d-v4"
model_saving_path = f"./Saves/{env_id}/"
learning_plot_saving_path = "Plots/"
replay_buffer_size = 1_000_000
critic_structure = [1024, 1024]
actor_structure = [1024, 1024]
total_steps = 5_000_000
batch_size = 64
gamma = 0.99
tau = 0.005
load = False
beta = 0.001
alpha = 0.001

# n_games = 10_000
max_steps = 10_000_000
warmup_steps = 1_000

##################################################
# DEMONSTRATION
##################################################

episode = 0
global_step = 0
env = gym.make(env_id, render_mode='human')
agent = Agent(alpha=alpha, beta=beta, model_path=model_saving_path, tau=tau, gamma=gamma,
              env=env, batch_size=batch_size, actor_architecture=actor_structure, cnt=400,
              critic_architecture=critic_structure, buffer_size=replay_buffer_size)

agent.load_models()

best_score = env.reward_range[0]
score_history = list()

while global_step < total_steps:
    observation = env.reset()[0]
    done = False
    score = 0

    while not done:
        global_step += 1

        action = agent.choose_action(observation)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)

        state = next_state
        score += reward
        if done:
            break

    score_history.append(score)

    print(f"Episode: {episode}, Reward: {score:.2f}, Step: {global_step}")
    episode += 1
