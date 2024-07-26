import numpy as np
import tensorflow as tf
from keras.layers import Dense
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential

def oom_solver():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

def soft_update(net, target_net, soft_tau=1e-2):
    net_weights = np.array(net.get_weights())
    target_net_weights = np.array(target_net.get_weights())
    updated_weights = soft_tau * target_net_weights + (1 - soft_tau) * net_weights
    target_net.set_weights(updated_weights)

def actor_network(num_inputs, num_actions, hidden_size):
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu'))    # , input_shape=num_inputs))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(num_actions[0], activation='tanh'))
    model.compile(optimizer=Adam())
    # model.summary()

    return model

def critic_network(num_inputs, num_actions, hidden_size):
    shape = (num_inputs[0] + num_actions[0],)
    model = Sequential()
    model.add(Dense(hidden_size, activation='relu', input_shape=shape))
    model.add(Dense(hidden_size, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer=Adam())
    model.summary()

    return model

def plot(frame_idx, rewards):
    plt.figure(figsize=(20, 5))
    plt.subplot(131)
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.plot(rewards)
    plt.show()
