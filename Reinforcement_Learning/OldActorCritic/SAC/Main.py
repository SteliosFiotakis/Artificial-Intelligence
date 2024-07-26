import gym
import numpy as np
from Agent import Agent
from utils import plot_learning_curve
from gym.wrappers import RescaleAction

##################################################
# HYPERPARAMETERS
##################################################

# TODO: make the main file the same.
# TODO: basically just change the algorith
# TODO: for each case.

env_id = "InvertedPendulum-v4"
folder_for_saves = "Saves"
folder_for_plot_save = "Plots"

load_checkpoint = False

# n_games = 250
total_training_steps = 1_000_000

##################################################
# INITIALIZATION
##################################################

base_env = gym.make(env_id)
env = RescaleAction(base_env, min_action=-1, max_action=1)

agent = Agent(environment=env)
# agent = Agent(input_dims=env.observation_space.shape, env=env,
#         n_actions=env.action_space.shape[0])

model_saving_path = f"./{folder_for_saves}/{env_id}/"
learning_plot_saving_path = f"./{folder_for_plot_save}/"

best_score = env.reward_range[0]
score_history = list()

##################################################
# TRAINING
##################################################

if load_checkpoint:
    agent.load_models()
    env.render()

step = 0
episode = 0

while step < total_training_steps:
    observation = env.reset()[0]
    done = False
    score = 0
    while not done:
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)
        agent.remember(observation, action, reward, observation_, done)
        observation = observation_
        score += reward
        agent.learn()
        step += 1
    score_history.append(score)
    avg_score = np.mean(score_history[-100:])

    if avg_score > best_score:
        best_score = avg_score
        agent.save_models()

    # print('episode ', i, 'score %.1f' % score, 'avg_score %.1f' % avg_score)
    print(f'Episode: {episode:_}, Score: {score:.2f}, Average score: {avg_score:.2f}, Step: {step:_}')
    episode += 1


x = [i+1 for i in range(episode)]
filename = learning_plot_saving_path + env_id + ".png"
plot_learning_curve(x, score_history, filename)
