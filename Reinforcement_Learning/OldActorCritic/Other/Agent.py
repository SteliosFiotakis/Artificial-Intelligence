import math
import random
import numpy as np
import tensorflow as tf
from HelperClasses import ReplayBuffer
from tensorflow import transpose as tft
from HelperFunctions import actor_network, critic_network, soft_update

# class cAgent:
#     def __init__(self, state_dimens, n_actions, es=1, ef=0.01, ed=500,
#                  memory_size=1000, gamma=0.99, number_of_memories=32, load_weights=False):
#         self.current_model = create_cnn(state_dimens, n_actions)
#         self.target_model = create_cnn(state_dimens, n_actions)
#         self.replay_buffer = ReplayBuffer(memory_size)
#
#         self.state_shape = state_dimens
#         self.action_shape = n_actions
#
#         self.epsilon = None
#         self.epsilon_start = es
#         self.epsilon_final = ef
#         self.epsilon_decay = ed
#
#         self.time_step = 0
#         self.gamma = gamma
#         self.number_of_memories = number_of_memories
#
#         if load_weights:
#             self.load()
#
#     def update_target(self):
#         self.target_model.set_weights(self.current_model.get_weights())
#
#     def update_epsilon(self):
#         self.time_step += 1
#         return self.epsilon_final + (self.epsilon_start - self.epsilon_final) * \
#             math.exp(-1. * self.time_step / self.epsilon_decay)
#
#     def take_action_e(self, state):
#         self.epsilon = self.update_epsilon()
#         if random.random() > self.epsilon:
#             q_values = self.current_model(state)
#             action = np.argmax(q_values)
#         else:
#             action = random.randrange(self.action_shape)
#         return action
#
#     def take_action(self, state):
#         q_values = self.current_model(state)
#         action = np.argmax(q_values)
#         return action
#
#     def remember(self, state, action, reward, next_state, termination):
#         self.replay_buffer.store(state, action, reward, next_state, termination)
#
#     def learn(self):
#         with tf.GradientTape() as tape:
#             state, action, reward, next_state, termination = self.replay_buffer.sample(self.number_of_memories)
#             action = list(action)
#             termination = np.array(termination)
#
#             for idx, value in enumerate(action):
#                 action[idx] = [idx, value]
#
#             q_values = self.current_model(state)             # current model's q_values for each action in current state
#             next_q_values = self.current_model(next_state)   # current model's q_values for each action in next state
#             next_q_target_values = self.target_model(next_state)  # target model's q_values for each action in nextstate
#
#             q_value = tf.gather_nd(q_values, action)    # the q_value for the action taken
#             # next_q_value = tf.reduce_max(next_q_values, axis=[1])
#             next_q_value = tf.math.argmax(next_q_values, axis=1)    # the q_value for the next state
#             next_q_value = next_q_value.numpy().tolist()
#
#             for idx, value in enumerate(next_q_value):
#                 next_q_value[idx] = [idx, value]
#
#             next_q_value = tf.gather_nd(next_q_target_values, next_q_value)
#             expected_q_value = reward + self.gamma * next_q_value * (1 - termination)
#
#             loss = (q_value - expected_q_value) ** 2
#
#         gradients = tape.gradient(loss, self.current_model.trainable_variables)
#
#         self.current_model.optimizer.apply_gradients(zip(gradients, self.current_model.trainable_variables))
#
#         return loss
#
#     def save(self):
#         with open("./Saves/WholeModels/Indexer.txt", "r+") as f:
#             number = f.read()
#             value_m = str(int(number) + 1)
#             f.seek(0)
#             f.truncate()
#             f.write(value_m)
#         self.current_model.save(f"./Saves/WholeModels/CurrentModel_{value_m}.h5")
#         self.target_model.save(f"./Saves/WholeModels/TargetModel_{value_m}.h5")
#         # with open("./Saves/WeightsOfModels/IndexerWeights.txt", "r+") as f:
#         #     number = f.read()
#         #     value_w = str(int(number) + 1)
#         #     f.seek(0)
#         #     f.truncate()
#         #     f.write(value_w)
#         # self.model.save_weights(f"./Saves/WeightsOfModels/Weights_{value_w}.h5")
#
#     def load(self):
#         # with open("./Saves/tempW/IndexerWeights.txt", "r") as f:
#         #     value = f.read()
#         # self.model.load_weights("./Saves/tempW/Weights_53")
#
#         with open("./Saves/WholeModels/Indexer.txt", "r") as f:
#             value = f.read()
#         self.current_model = tf.keras.models.load_model(f"./Saves/WholeModels/CurrentModel_{value}.h5")
#         self.target_model = tf.keras.models.load_model(f"./Saves/WholeModels/TargetModel_{value}.h5")
#
#         # self.model = tf.keras.models.load_model(f"./trainNetworkInEPS26.h5")
#         # self.model.summary()

class Agent:
    def __init__(self, replay_buffer_size=10_000, batch_size=32, load_weights=False,
                 num_inputs=3, num_actions=1, hidden_size=128, gamma=0.99):
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
        self.actor = actor_network(num_inputs, num_actions, hidden_size)
        self.critic = critic_network(num_inputs, num_actions, hidden_size)
        self.target_actor = actor_network(num_inputs, num_actions, hidden_size)
        self.target_critic = critic_network(num_inputs, num_actions, hidden_size)

        if load_weights:
            self.load()

    def remember(self, state, action, reward, next_state, truncated):
        self.replay_buffer.push(state, action, reward, next_state, truncated)

    def act(self, state):
        return self.actor(state)

    def learn(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
        done = np.array(done)

        with tf.GradientTape() as actor_tape:
            actor_loss = self.critic(tf.concat([state, self.actor(state)], 1))
            actor_loss = -(tf.reduce_mean(actor_loss))
            # actor_loss = -actor_loss.mean()

        with tf.GradientTape() as critic_tape:
            next_action = self.target_actor(next_state)
            target_value = self.target_critic(tf.concat([next_state, next_action], 1))
            expected_value = reward + (1.0 - done) * self.gamma * target_value
            # expected_value = torch.clamp(expected_value, min_value, max_value)

            value = self.critic(tf.concat([state, action], 1))
            value_loss = (value - expected_value) ** 2
            # value_loss = value_criterion(value, expected_value.detach())

        actor_gradients = actor_tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))

        critic_gradients = critic_tape.gradient(value_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))

        soft_update(self.actor, self.target_actor)
        soft_update(self.critic, self.target_critic)

    def save(self):
        with open("Saves/Pendulum/Indexer.txt", "r+") as f:
            number = f.read()
            value_m = str(int(number) + 1)
            f.seek(0)
            f.truncate()
            f.write(value_m)
        self.actor.save(f"./Saves/Pendulum/Actor_{value_m}.h5")
        self.critic.save(f"./Saves/Pendulum/Critic_{value_m}.h5")
        self.target_actor.save(f"./Saves/Pendulum/TargetActor_{value_m}.h5")
        self.target_critic.save(f"./Saves/Pendulum/TargetCritic_{value_m}.h5")

    def load(self):
        with open("Saves/Pendulum/Indexer.txt", "r") as f:
            value = f.read()
        self.actor = tf.keras.models.load_model(f"./Saves/Pendulum/Actor_{value}.h5")
        self.critic = tf.keras.models.load_model(f"./Saves/Pendulum/Critic_{value}.h5")
        self.target_actor = tf.keras.models.load_model(f"./Saves/Pendulum/TargetActor_{value}.h5")
        self.target_critic = tf.keras.models.load_model(f"./Saves/Pendulum/TargetCritic_{value}.h5")
