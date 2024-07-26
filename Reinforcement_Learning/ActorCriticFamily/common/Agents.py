"""
All Agents have some common capabilities and functionalities.

- Their core meaning is, given a state, choosing an action to give back.

- Most algoriths implements a memory capability, so the agent needs the corresponding functions.

- All Agents have some hyperparameters like "gamma" or file paths for saves and other things.

- A function for saving agent's configuration.

- A function for loading agent's configuration.
"""

import numpy as np
import tensorflow as tf
from keras.losses import MSE
from keras.optimizers import Adam
from ActorCriticFamily.common.ReplayBuffers import ReplayBufferNP
from ActorCriticFamily.common.NeuralNetworks import (actor_network, actor_with_prob, action_and_prob_from_actor,
                                                     critic_network, value_network)

class BaseAgent:
    """
    The base class for an Agent.
    """
    def __init__(self,
                 buffer_size=1_000_000, batch_size=64,
                 noise=0.001, alpha=0.0003, beta=0.0003,
                 models_saving_path="", model_loading_path="",
                 env=None, gamma=0.99, tau=0.005, cnt=-1, models_format='tf'):

        self.state_dims = env.observation_space.shape[0]
        self.action_dims = env.action_space.shape[0]

        self.gamma = gamma
        self.tau = tau
        self.cnt = cnt

        self.alpha = alpha
        self.beta = beta

        self.save_path = models_saving_path
        self.load_path = model_loading_path
        self.save_format = models_format

        self.memory = ReplayBufferNP(buffer_size, tuple([self.state_dims]), self.action_dims)
        self.batch_size = batch_size

        self.networks_list = list()

        self.learn_step_cnt = 0

        self.noise = noise

    def remember(self, state, action, reward, next_state, termination):
        self.memory.store(state, action, reward, next_state, termination)

    def save(self):
        self.cnt += 1
        print(f'Saving... #{self.cnt}')
        for network in self.networks_list:
            network.save_weights(f"{self.save_path}/{network.model_name}_{str(self.cnt)}", save_format=self.save_format)

    def load(self, cnt):
        print('... loading models ...')
        for network in self.networks_list:
            network.load_weights(f"{self.save_path}/{network.model_name}_{str(cnt)}")

class DDPG(BaseAgent):
    """
    Agent with DDPG algorithm.
    """
    def __init__(self, actor_structure=None, target_actor_structure=None,
                 critic_structure=None, target_critic_structure=None, **kwargs):
        super().__init__(**kwargs)

        self.actor = actor_network(tuple([self.state_dims]), self.action_dims, actor_structure, name="Actor")
        self.target_actor = actor_network(tuple([self.state_dims]), self.action_dims, target_actor_structure, name="Target_Actor")
        self.critic = critic_network(tuple([self.state_dims + self.action_dims]), critic_structure, name="Critic")
        self.target_critic = critic_network(tuple([self.state_dims + self.action_dims]), target_critic_structure, name="Target_Critic")

        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_critic.compile(optimizer=Adam(learning_rate=self.beta))

        self.networks_list = [self.actor, self.target_actor, self.critic, self.target_critic]

        self.soft_update(tau=1)

        self.noise = 0.1

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        actions += tf.random.normal(shape=[self.action_dims], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, -1, 1)

        return actions[0]

    def learn(self):
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        done = np.array(done)

        state = tf.convert_to_tensor(state, dtype=tf.float32)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)

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

        self.soft_update(self.tau)

    def soft_update(self, tau=None):
        actor_weights = np.array(self.actor.get_weights(), dtype=object)
        target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
        updated_weights = tau * target_actor_weights + (1 - tau) * actor_weights
        self.target_actor.set_weights(updated_weights)

        critic_weights = np.array(self.critic.get_weights(), dtype=object)
        target_critic_weights = np.array(self.target_critic.get_weights(), dtype=object)
        updated_weights = tau * target_critic_weights + (1 - tau) * critic_weights
        self.target_critic.set_weights(updated_weights)

class TD3(BaseAgent):
    """
    Agent with TD3 algorithm.
    """
    def __init__(self, actor_structure=None, target_actor_structure=None,
                 critic_1_structure=None, target_critic_1_structure=None,
                 critic_2_structure=None, target_critic_2_structure=None, **kwargs):
        super().__init__(**kwargs)

        self.actor = actor_network(tuple([self.state_dims]), self.action_dims, actor_structure, name="Actor")
        self.target_actor = actor_network(tuple([self.state_dims]), self.action_dims, target_actor_structure, name="Target_Actor")
        self.critic_1 = critic_network(tuple([self.state_dims + self.action_dims]), critic_1_structure, name="Critic_1")
        self.target_critic_1 = critic_network(tuple([self.state_dims + self.action_dims]), target_critic_1_structure, name="Target_Critic_1")
        self.critic_2 = critic_network(tuple([self.state_dims + self.action_dims]), critic_2_structure, name="Critic_2")
        self.target_critic_2 = critic_network(tuple([self.state_dims + self.action_dims]), target_critic_2_structure, name="Target_Critic_2")

        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.target_actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_critic_1.compile(optimizer=Adam(learning_rate=self.beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_critic_2.compile(optimizer=Adam(learning_rate=self.beta))

        self.networks_list = [self.actor, self.target_actor, self.critic_1,
                              self.target_critic_1, self.critic_2, self.target_critic_2]

        self.update_network_parameters(tau=1)

        self.noise = 0.1
        self.update_actor_iter = 2

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, -1, 1)

        return mu_prime

    def learn(self):
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, -1, 1)

            q1_ = self.target_critic_1(tf.concat([states_, target_actions], 1))
            q2_ = self.target_critic_2(tf.concat([states_, target_actions], 1))

            q1 = tf.squeeze(self.critic_1(tf.concat([states, actions], 1)), 1)
            q2 = tf.squeeze(self.critic_2(tf.concat([states, actions], 1)), 1)

            # shape is [number_of_memories, 1], want to collapse to [number_of_memories]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma * critic_value_ * (1 - dones)
            critic_1_loss = MSE(target, q1)
            critic_2_loss = MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cnt += 1

        if self.learn_step_cnt % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(tf.concat([states, new_actions], 1))
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)

class PPO(BaseAgent):
    """
    Agent with PPO algorithm.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        # returns a batch size of 1, want a scalar array
        mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        return mu_prime

    def learn(self):
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as tape:
            target_actions = self.target_actor(states_)
            target_actions = target_actions + tf.clip_by_value(np.random.normal(scale=0.2), -0.5, 0.5)

            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            q1_ = self.target_critic_1(tf.concat([states_, target_actions], 1))
            q2_ = self.target_critic_2(tf.concat([states_, target_actions], 1))

            q1 = tf.squeeze(self.critic_1(tf.concat([states, actions], 1)), 1)
            q2 = tf.squeeze(self.critic_2(tf.concat([states, actions], 1)), 1)

            # shape is [number_of_memories, 1], want to collapse to [number_of_memories]
            q1_ = tf.squeeze(q1_, 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            # in tf2 only integer scalar arrays can be used as indices
            # and eager exection doesn't support assignment, so we can't do
            # q1_[dones] = 0.0
            target = rewards + self.gamma * critic_value_ * (1 - dones)
            critic_1_loss = MSE(target, q1)
            critic_2_loss = MSE(target, q2)

        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_step_cnt += 1

        if self.learn_step_cnt % self.update_actor_iter != 0:
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(tf.concat([states, new_actions], 1))
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_1.set_weights(weights)

        weights = []
        targets = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_critic_2.set_weights(weights)

class SAC(BaseAgent):
    """
    Agent with SAC algorithm.
    """
    def __init__(self, actor_structure=None, critic_1_structure=None, critic_2_structure=None,
                 value_structure=None, target_value_structure=None, **kwargs):
        super().__init__(**kwargs)

        self.actor = actor_with_prob(tuple([self.state_dims]), self.action_dims, actor_structure, name="Actor")
        self.critic_1 = critic_network(tuple([self.state_dims + self.action_dims]), critic_1_structure, name="Critic_1")
        self.critic_2 = critic_network(tuple([self.state_dims + self.action_dims]), critic_2_structure, name="Critic_2")
        self.value = value_network(tuple([self.state_dims]), value_structure, name="Value")
        self.target_value = value_network(tuple([self.state_dims]), target_value_structure, name="Target_Value")

        self.actor.compile(optimizer=Adam(learning_rate=self.alpha))
        self.critic_1.compile(optimizer=Adam(learning_rate=self.beta))
        self.critic_2.compile(optimizer=Adam(learning_rate=self.beta))
        self.value.compile(optimizer=Adam(learning_rate=self.beta))
        self.target_value.compile(optimizer=Adam(learning_rate=self.beta))

        self.networks_list = [self.actor, self.critic_1, self.critic_2, self.value, self.target_value]

        self.update_network_parameters(tau=1)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions, _ = action_and_prob_from_actor(self.actor(state))

        return actions[0]

    def learn(self):
        state, action, reward, new_state, done = self.memory.sample(self.batch_size)

        states = tf.convert_to_tensor(state, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_state, dtype=tf.float32)
        # rewards = tf.convert_to_tensor(reward, dtype=tf.float32)
        # actions = tf.convert_to_tensor(action, dtype=tf.float32)

        with tf.GradientTape() as tape:
            value = tf.squeeze(self.value(states), 1)
            value_ = tf.squeeze(self.target_value(states_), 1)

            current_policy_actions, log_probs = action_and_prob_from_actor(self.actor(states))
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(tf.concat([states, current_policy_actions], 1))
            q2_new_policy = self.critic_2(tf.concat([states, current_policy_actions], 1))
            critic_value = tf.squeeze(tf.math.minimum(q1_new_policy, q2_new_policy), 1)

            value_target = critic_value - log_probs
            value_loss = 0.5 * MSE(value, value_target)

        value_network_gradient = tape.gradient(value_loss, self.value.trainable_variables)
        self.value.optimizer.apply_gradients(zip(value_network_gradient, self.value.trainable_variables))

        with tf.GradientTape() as tape:
            # in the original paper, they reparameterize here. We don't implement
            # this, so it's just the usual action.
            new_policy_actions, log_probs = action_and_prob_from_actor(self.actor(states))
            log_probs = tf.squeeze(log_probs, 1)
            q1_new_policy = self.critic_1(tf.concat([states, new_policy_actions], 1))
            q2_new_policy = self.critic_2(tf.concat([states, new_policy_actions], 1))
            critic_value = tf.squeeze(tf.math.minimum(
                q1_new_policy, q2_new_policy), 1)

            actor_loss = log_probs - critic_value
            actor_loss = tf.math.reduce_mean(actor_loss)

        actor_network_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_network_gradient, self.actor.trainable_variables))

        with tf.GradientTape(persistent=True) as tape:
            # I didn't know that these context managers shared values?
            q_hat = reward + self.gamma * value_ * (1 - done)
            q1_old_policy = tf.squeeze(self.critic_1(tf.concat([state, action], 1)), 1)
            q2_old_policy = tf.squeeze(self.critic_2(tf.concat([state, action], 1)), 1)
            critic_1_loss = 0.5 * MSE(q1_old_policy, q_hat)
            critic_2_loss = 0.5 * MSE(q2_old_policy, q_hat)

        critic_1_network_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_network_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_network_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_network_gradient, self.critic_2.trainable_variables))

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_value.weights
        for i, weight in enumerate(self.value.weights):
            weights.append(weight * tau + targets[i]*(1-tau))

        self.target_value.set_weights(weights)
