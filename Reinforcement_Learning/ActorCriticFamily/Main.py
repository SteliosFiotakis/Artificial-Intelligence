import gym
from gym.wrappers import RescaleAction

from configparser import ConfigParser

from common import Agents
from common.Utils import save_learning_plot

import numpy as np
import os

file = "Config.ini"

config = ConfigParser()
config.read(file)

demonstrate = eval(config['High-Level HyperParameters']['demonstrate'])

##################################################
# ENVIRONMENT INITIALIZATION
##################################################

env_id = config['High-Level HyperParameters']['environment_name']
if demonstrate:
    base_env = gym.make(env_id, render_mode='human')
else:
    base_env = gym.make(env_id, forward_reward_weight=10.0)
env = RescaleAction(base_env, min_action=-1, max_action=1)

##################################################
# PATHS INITIALIZATION
##################################################

path_for_models_saving = config['High-Level HyperParameters']['path_for_models_saving']
path_for_model_loading = config['High-Level HyperParameters']['path_for_model_loading']
path_for_plots_saving = config['High-Level HyperParameters']['path_for_plots_saving']

plots_file_format = config['Advanced Settings']['plots_file_format']
plots_name_prefix = config['Advanced Settings']['plots_name_prefix']

plots_name = f"{path_for_plots_saving}/{plots_name_prefix}.{plots_file_format}"

##################################################
# PATHS EXISTANCE VALIDATION
##################################################

if not os.path.exists(path_for_models_saving):
    os.makedirs(path_for_models_saving)

if not os.path.exists(path_for_model_loading):
    os.makedirs(path_for_model_loading)

if not os.path.exists(path_for_plots_saving):
    os.makedirs(path_for_plots_saving)

##################################################
# ALGORITHM AND AGENT INITIALIZATION
##################################################

batch_size = eval(config['Low-Level HyperParameters']['batch_size'])

algorithm = config['High-Level HyperParameters']['algorithm_to_be_used']
default_hyperparameters = eval(config['High-Level HyperParameters']['default_hyperparameters'])

models_saving_format = config['Low-Level HyperParameters']['models_saving_format']

if algorithm == "DDPG":
    if default_hyperparameters:
        agent = Agents.DDPG(env=env)
    else:
        agent = Agents.DDPG(env=env)
elif algorithm == "TD3":

    actor_structure = eval(config['TD3 HyperParameters']['actor_structure'])
    critic_1_structure = eval(config['TD3 HyperParameters']['critic_1_structure'])
    critic_2_structure = eval(config['TD3 HyperParameters']['critic_2_structure'])
    target_actor_structure = eval(config['TD3 HyperParameters']['target_actor_structure'])
    target_critic_1_structure = eval(config['TD3 HyperParameters']['target_critic_1_structure'])
    target_critic_2_structure = eval(config['TD3 HyperParameters']['target_critic_2_structure'])

    if default_hyperparameters:
        agent = Agents.TD3(env=env, batch_size=batch_size,
                           models_saving_path=path_for_models_saving,
                           model_loading_path=path_for_model_loading,
                           actor_structure=actor_structure,
                           critic_1_structure=critic_1_structure,
                           critic_2_structure=critic_2_structure,
                           target_actor_structure=target_actor_structure,
                           target_critic_1_structure=target_critic_1_structure,
                           target_critic_2_structure=target_critic_2_structure,
                           models_format=models_saving_format)
    else:
        agent = Agents.DDPG(env=env)
elif algorithm == "PPO":
    if default_hyperparameters:
        agent = Agents.PPO(env=env)
    else:
        agent = Agents.DDPG(env=env)
elif algorithm == "SAC":

    actor_structure = eval(config['SAC HyperParameters']['actor_structure'])
    critic_1_structure = eval(config['SAC HyperParameters']['critic_1_structure'])
    critic_2_structure = eval(config['SAC HyperParameters']['critic_2_structure'])
    value_structure = eval(config['SAC HyperParameters']['value_structure'])
    target_value_structure = eval(config['SAC HyperParameters']['target_value_structure'])

    if default_hyperparameters:
        agent = Agents.SAC(env=env, batch_size=batch_size,
                           models_saving_path=path_for_models_saving,
                           model_loading_path=path_for_model_loading,
                           actor_structure=actor_structure,
                           critic_1_structure=critic_1_structure,
                           critic_2_structure=critic_2_structure,
                           value_structure=value_structure,
                           target_value_structure=target_value_structure,
                           models_format=models_saving_format)
    else:
        agent = Agents.DDPG(env=env)
else: raise ValueError("Wrong algorithm name.")

##################################################
# TRAINING PARAMETERS INITIALIZATION
##################################################

episodes_until_saving_plot = eval(config['High-Level HyperParameters']['episodes_until_saving_plot'])

total_training_steps = eval(config['High-Level HyperParameters']['total_training_step'])

load_checkpoint = eval(config['High-Level HyperParameters']['load_existing_models'])

if load_checkpoint or demonstrate:
    last_cnt_save = eval(config['High-Level HyperParameters']['last_cnt_save'])
    agent.load(last_cnt_save)
    if load_checkpoint:
        with open("arrays_save.txt", "r") as file:
            episode = eval(file.readline())
            best_score = eval(file.readline())
            score_history = eval(file.readline())
            average_score_history = eval(file.readline())
else:
    best_score = env.reward_range[0]
    average_score_history = list()
    score_history = list()
    episode = 0
step = 0

##################################################
# WARMUP
##################################################

warmup_steps = eval(config['Low-Level HyperParameters']['warmup_steps'])

if not demonstrate:
    observation = env.reset()[0]
    for _ in range(warmup_steps + batch_size):
        action = env.action_space.sample()
        observation_, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)
        agent.remember(observation, action, reward, observation_, terminated)
        if done:
            observation = env.reset()[0]
        else:
            observation = observation_

##################################################
# DEMONSTRATION
##################################################

if demonstrate:
    observation = env.reset()[0]
    score = 0
    for _ in range(warmup_steps + batch_size):
        action = agent.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)
        score += reward
        if done:
            observation = env.reset()[0]
            print(f'Score: {score:_.2f}')
            score = 0
        else:
            observation = observation_
    exit()

##################################################
# TRAINING
##################################################

try:
    while step < total_training_steps:
        observation = env.reset()[0]
        done = False
        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = max(terminated, truncated)
            agent.remember(observation, action, reward, next_observation, terminated)
            observation = next_observation
            score += reward
            agent.learn()
            step += 1
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        average_score_history.append(avg_score)

        if avg_score > best_score:
            best_score = avg_score
            agent.save()

        episode += 1
        print(f'Episode: {episode:_}, Score: {score:_.2f}, Average score: {avg_score:_.2f}, Step: {step:_}')

        if episode % episodes_until_saving_plot == 0:
            x = [i for i in range(episode)]
            save_learning_plot(x, average_score_history, plots_name)
except KeyboardInterrupt:
    with open("arrays_save.txt", "w") as file:
        file.write(str(episode))
        file.write("\n")
        file.write(str(best_score))
        file.write("\n")
        file.write(str(score_history))
        file.write("\n")
        file.write(str(average_score_history))
