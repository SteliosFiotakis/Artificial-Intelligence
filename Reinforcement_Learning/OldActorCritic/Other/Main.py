##################################################
# IMPORTS
##################################################

import gym
import numpy as np
import tensorflow as tf
from Agent import Agent
from HelperFunctions import oom_solver, plot
from HelperClasses import NormalizedActions, OUNoise

##################################################
# OOM SOLVER
##################################################

oom_solver()

##################################################
# HYPERPARAMETERS
##################################################

env_id = "Pendulum-v1"
training = False
load = True
memory = 50_000
gamma = 0.99
plotting_time = 100

replay_buffer_size = 1_000_000
total_steps = 400
max_steps = 500
rewards = list()
batch_size = 128

##################################################
# Initialization
##################################################

env = gym.make(env_id)
# env = NormalizedActions(env)
ou_noise = OUNoise(env.action_space)

state_dim = env.observation_space.shape
action_dim = env.action_space.shape
hidden_dim = 256

agent = Agent(hidden_size=hidden_dim,
              num_actions=action_dim,
              batch_size=batch_size,
              num_inputs=state_dim,
              load_weights=load,
              replay_buffer_size=replay_buffer_size)

##################################################
# TRAINING
##################################################

episode = 0
step_idx = 0
model_cnt = 0
best_score = env.reward_range[0]
average_rewards = env.reward_range[0]

if training:
    while step_idx < total_steps:
        state = tf.convert_to_tensor([env.reset()[0]])
        ou_noise.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.act(state)
            action = ou_noise.get_action(action, step)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.transpose(next_state)
            next_state = tf.convert_to_tensor(next_state)

            agent.remember(state, action, reward, next_state, truncated)

            if step_idx > batch_size:
                agent.learn()

            state = next_state
            episode_reward += reward
            step_idx += 1

            if truncated:
                print(f"Episode: {episode}, Reward: {episode_reward}, Step: {step_idx}")
                episode += 1
                average_rewards = np.mean(rewards[-50:])

                if average_rewards > best_score:
                    best_score = average_rewards
                    print(f"Saving Model... [{model_cnt}]")
                    model_cnt += 1
                    # agent.save()

                break

        rewards.append(episode_reward)

##################################################
# DEMONSTRATION
##################################################

# plot(step_idx, rewards)

episode = 0
step_idx = 0
dem_tries = 1_000

env = gym.make(env_id, render_mode="human")

while step_idx < dem_tries:
    state = tf.convert_to_tensor([env.reset()[0]])
    ou_noise.reset()
    episode_reward = 0

    for step in range(max_steps):
        action = agent.act(state)
        action = ou_noise.get_action(action, step)

        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = np.transpose(next_state)
        next_state = tf.convert_to_tensor(next_state)

        state = next_state
        episode_reward += reward
        step_idx += 1

        if truncated:
            print(f"Episode: {episode}, Reward: {episode_reward}, Step: {step_idx}")
            episode += 1
            break

    rewards.append(episode_reward)

env.close()
