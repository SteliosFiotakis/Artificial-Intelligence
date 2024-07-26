##################################################
# IMPORTS
##################################################

import gym
import math
import random
import numpy as np
import tensorflow as tf
from collections import deque
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential

##################################################
# OOM SOLVER
##################################################

def oom_solver():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

##################################################
# REPLAY_BUFFER
##################################################

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, cstate, caction, creward, cnext_state, cdone):
        self.buffer.append((cstate, caction, creward, cnext_state, cdone))

    def sample(self, cbatch_size):
        cstate, caction, creward, cnext_state, cdone = zip(*random.sample(self.buffer, cbatch_size))
        return np.concatenate(cstate), caction, creward, np.concatenate(cnext_state), cdone

    def __len__(self):
        return len(self.buffer)

##################################################
# HYPERPARAMETERS
##################################################

replay_buffer = ReplayBuffer(1000)

env_id = "CartPole-v1"
env = gym.make(env_id)

epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay = 500


def calculate_epsilon(step_n):
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * step_n / epsilon_decay)


def compute_td_loss(fbatch_size):
    with tf.GradientTape() as tape:
        fstate, faction, freward, fnext_state, fdone = replay_buffer.sample(fbatch_size)
        faction = list(faction)
        fdone = np.array(fdone)

        fq_values = model(fstate)
        next_q_values = model(fnext_state)

        for idx, value in enumerate(faction):
            faction[idx] = [idx, value]

        q_value = tf.gather_nd(fq_values, faction)
        next_q_value = tf.reduce_max(next_q_values, axis=[1])
        expected_q_value = freward + gamma * next_q_value * (1 - fdone)

        floss = (q_value - expected_q_value) ** 2

    gradients = tape.gradient(floss, model.trainable_variables)

    model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return floss


def plot(step_idx, frewards, flosses):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (step_idx, np.mean(frewards[-10:])))
    plt.plot(frewards)
    plt.subplot(132)
    plt.title('loss')
    plt.plot(flosses)
    plt.show()


##################################################
# MODEL CREATION
##################################################

model = Sequential()
model.add(Dense(128, activation='relu', input_shape=env.observation_space.shape))
model.add(Dense(128, activation='relu'))
model.add(Dense(env.action_space.n, activation=None))
model.compile(optimizer=Adam())
model.summary()

##################################################
# TRAINING
##################################################

max_steps = 100
batch_size = 32
gamma = 0.99

losses = []
all_rewards = []
episode_reward = 0

state = tf.convert_to_tensor([env.reset()[0]])
for step in range(1, max_steps + 1):
    epsilon = calculate_epsilon(step_n=step)
    if random.random() > epsilon:
        q_values = model(state)
        action = np.argmax(q_values)
    else:
        action = random.randrange(env.action_space.n)

    next_state, reward, terminated, truncated, info = env.step(action)
    done = max(terminated, truncated)
    next_state = tf.convert_to_tensor([next_state])
    replay_buffer.push(state, action, reward, next_state, done)

    state = next_state
    episode_reward += reward

    if done:
        state = tf.convert_to_tensor([env.reset()[0]])
        all_rewards.append(episode_reward)
        print(episode_reward)
        episode_reward = 0

    if len(replay_buffer) > batch_size:
        loss = compute_td_loss(batch_size)
        loss = tf.reduce_mean(loss)
        losses.append(loss)

    if step % 1_000 == 0:
        plot(step_idx=step, frewards=all_rewards, flosses=losses)

# demonstrate = 1000
# env = gym.make("CartPole-v1", render_mode='human')
#
# state = tf.convert_to_tensor([env.reset()[0]])
# for step in range(1, demonstrate + 1):
#     q_values = model(state)
#     action = np.argmax(q_values)
#
#     next_state, reward, terminated, truncated, info = env.step(action)
#     termination = max(terminated, truncated)
#     next_state = tf.convert_to_tensor([next_state])
#
#     state = next_state
#
#     if termination:
#         state = tf.convert_to_tensor([env.reset()[0]])
