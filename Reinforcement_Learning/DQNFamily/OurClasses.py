import math
import random
import numpy as np
import tensorflow as tf
from collections import deque
from OurFunctions import create_cnn, create_nn


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


class Agent:
    def __init__(self, state_dimens, n_actions, es=1, ef=0.01, ed=500,
                 memory_size=1000, gamma=0.99, batch_size=32, load_weights=False):
        self.current_model = create_nn(state_dimens, n_actions)
        self.target_model = create_nn(state_dimens, n_actions)
        self.replay_buffer = ReplayBuffer(memory_size)

        self.state_shape = state_dimens
        self.action_shape = n_actions

        self.epsilon = None
        self.epsilon_start = es
        self.epsilon_final = ef
        self.epsilon_decay = ed

        self.time_step = 0
        self.gamma = gamma
        self.batch_size = batch_size

        if load_weights:
            self.load()

    def update_target(self):
        self.target_model.set_weights(self.current_model.get_weights())

    def update_epsilon(self):
        self.time_step += 1
        return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
            math.exp(-1. * self.time_step / self.epsilon_decay)

    def take_action_e(self, observation):
        self.epsilon = self.update_epsilon()
        if random.random() > self.epsilon:
            q_values = self.current_model(observation)
            action = np.argmax(q_values)
        else:
            action = random.randrange(self.action_shape)
        return action

    def take_action(self, observation):
        q_values = self.current_model(observation)
        action = np.argmax(q_values)
        return action

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self):
        with tf.GradientTape() as tape:
            state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            action = list(action)
            done = np.array(done)

            for idx, value in enumerate(action):
                action[idx] = [idx, value]

            q_values = self.current_model(state)             # current model's q_values for each action in current state
            next_q_values = self.current_model(next_state)   # current model's q_values for each action in next state
            next_q_target_values = self.target_model(next_state)  # target model's q_values for each action in nextstate

            q_value = tf.gather_nd(q_values, action)    # the q_value for the action taken
            # next_q_value = tf.reduce_max(next_q_values, axis=[1])
            next_q_value = tf.math.argmax(next_q_values, axis=1)    # the q_value for the next state
            next_q_value = next_q_value.numpy().tolist()

            for idx, value in enumerate(next_q_value):
                next_q_value[idx] = [idx, value]

            next_q_value = tf.gather_nd(next_q_target_values, next_q_value)
            expected_q_value = reward + self.gamma * next_q_value * (1 - done)

            loss = (q_value - expected_q_value) ** 2

        gradients = tape.gradient(loss, self.current_model.trainable_variables)

        self.current_model.optimizer.apply_gradients(zip(gradients, self.current_model.trainable_variables))

        return loss

    def save(self):
        with open("./Saves/WholeModels/Indexer.txt", "r+") as f:
            number = f.read()
            value_m = str(int(number) + 1)
            f.seek(0)
            f.truncate()
            f.write(value_m)
        self.current_model.save(f"./Saves/WholeModels/CurrentModel_{value_m}.h5")
        self.target_model.save(f"./Saves/WholeModels/TargetModel_{value_m}.h5")
        # with open("./Saves/WeightsOfModels/IndexerWeights.txt", "r+") as f:
        #     number = f.read()
        #     value_w = str(int(number) + 1)
        #     f.seek(0)
        #     f.truncate()
        #     f.write(value_w)
        # self.model.save_weights(f"./Saves/WeightsOfModels/Weights_{value_w}.h5")

    def load(self):
        # with open("./Saves/tempW/IndexerWeights.txt", "r") as f:
        #     value = f.read()
        # self.model.load_weights("./Saves/tempW/Weights_53")

        with open("./Saves/CartPole/Indexer.txt", "r") as f:
            value = f.read()
        self.current_model = tf.keras.models.load_model(f"./Saves/CartPole/CurrentModel_{value}.h5")
        self.target_model = tf.keras.models.load_model(f"./Saves/CartPole/TargetModel_{value}.h5")

        # self.model = tf.keras.models.load_model(f"./trainNetworkInEPS26.h5")
        # self.model.summary()
