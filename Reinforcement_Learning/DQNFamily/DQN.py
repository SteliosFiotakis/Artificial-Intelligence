##################################################
# IMPORTS
##################################################

import gym
import csv
import numpy as np
import tensorflow as tf
from OurClasses import Agent
from OurFunctions import oom_solver

##################################################
# OOM SOLVER
##################################################

oom_solver()

##################################################
# HYPERPARAMETERS
##################################################

env_id = "Pong-v0"

training = True
load = False

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 100_000

batch_size = 32
memory = 50_000
gamma = 0.99

max_steps = 2_000_000
plotting_time = 100

env = gym.make(env_id)

agent = Agent(env.observation_space.shape, env.action_space.n, epsilon_start,
              epsilon_final, epsilon_decay, memory, gamma, batch_size, load)

##################################################
# TRAINING
##################################################

if training:

    episode = 0
    model_cnt = 0

    losses = list()
    all_rewards = list()
    episode_reward = 0
    best_score = env.reward_range[0]
    average_rewards = env.reward_range[0]

    state = tf.convert_to_tensor([env.reset()[0]])
    for step in range(1, max_steps + 1):
        action = agent.take_action_e(state)

        next_state, reward, terminated, truncated, info = env.step(action)
        print(terminated, truncated)
        done = max(terminated, truncated)
        next_state = tf.convert_to_tensor([next_state])
        # if terminated:
        #     reward = 10
        #     print('TERMINATED!!!')

        agent.remember(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            print(f"Episode: {episode}, Reward: {episode_reward}, Step: {step}, Epsilon: {agent.epsilon}")
            episode += 1
            state = tf.convert_to_tensor([env.reset()[0]])
            all_rewards.append(episode_reward)
            episode_reward = 0
            # if len(all_rewards) > 100:
            average_rewards = np.mean(all_rewards[-100:])

            if average_rewards > best_score:
                best_score = average_rewards
                model_cnt += 1
                print(f"Saving Model... [{model_cnt}]")
                agent.save()

        if step > batch_size:
            loss = agent.learn()
            loss = tf.reduce_mean(loss)
            losses.append(loss)

        if step % plotting_time == 0:
            # plot(step_idx=step, frewards=all_rewards, flosses=losses)
            with open('example.csv', 'w') as file:
                writer = csv.writer(file)
                data = [step, all_rewards]
                writer.writerow(data)
            agent.update_target()


##################################################
# DEMONSTRATION
##################################################

demonstration = 10_000
env = gym.make(env_id, render_mode='human')

state = tf.convert_to_tensor([env.reset()[0]])
for step in range(1, demonstration + 1):
    action = agent.take_action(state)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = max(terminated, truncated)
    next_state = tf.convert_to_tensor([next_state])

    state = next_state

    if done:
        state = tf.convert_to_tensor([env.reset()[0]])
