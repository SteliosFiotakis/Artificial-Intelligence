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
# total_training_steps = 5_000_000
batch_size = 128
gamma = 0.99
tau = 0.005
load = False
beta = 0.001
alpha = 0.001

# n_games = 10_000
max_steps = 20_000_000
warmup_steps = 1_000

##################################################
# TRAINING
##################################################

# env = gym.make(env_id, terminate_when_unhealthy=False)
env = gym.make(env_id, render_mode='human')
agent = Agent(alpha=alpha, beta=beta, model_path=model_saving_path, tau=tau, gamma=gamma,
              env=env, batch_size=batch_size, actor_architecture=actor_structure, cnt=583,
              critic_architecture=critic_structure, buffer_size=replay_buffer_size)

best_score = env.reward_range[0]
score_history = list()

agent.load_models()

observation = env.reset()[0]
for step in range(warmup_steps + batch_size):
    action = env.action_space.sample()
    observation_, reward, terminated, truncated, info = env.step(action)
    done = max(terminated, truncated)
    agent.remember(observation, action, reward, observation_, done)
    if done:
        observation = env.reset()[0]
    else:
        observation = observation_

step = 0
episode = 0
while step < max_steps:
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        agent.remember(observation, action, reward, observation_, terminated)
        done = max(terminated, truncated)
        observation = observation_
        score += reward
        agent.learn()
        step += 1
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    print(f'Episode: {episode:_}, Score: {score:.2f}, Average score: {avg_score:.2f}, Step: {step:_}')
    episode += 1

x = [i+1 for i in range(episode)]
filename = learning_plot_saving_path + env_id + ".png"
plot_learning_curve(x, score_history, filename)
