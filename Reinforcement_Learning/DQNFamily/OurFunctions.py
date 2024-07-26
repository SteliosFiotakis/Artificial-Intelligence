import math
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Adam
from keras.models import Sequential


def oom_solver():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


def calculate_epsilon(step_n, eps_s, eps_e, eps_d):
    return eps_e + (eps_s - eps_e) * math.exp(-1. * step_n / eps_d)


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


def create_nn(state_dims, n_actions):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=state_dims))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer=Adam())
    model.summary()

    return model


def create_cnn(state_dims, n_actions):
    model = Sequential()
    model.add(Conv2D(16, activation='relu', kernel_size=8, strides=4, input_shape=state_dims))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(32, activation='relu', kernel_size=4, strides=2))
    model.add(MaxPooling2D(2))
    model.add(Conv2D(64, activation='relu', kernel_size=3, strides=1))
    model.add(MaxPooling2D(2))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(n_actions, activation='linear'))
    model.compile(optimizer=Adam())
    model.summary()

    return model
