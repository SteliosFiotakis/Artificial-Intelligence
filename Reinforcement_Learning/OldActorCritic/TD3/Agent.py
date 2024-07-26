import numpy as np
import tensorflow as tf
from keras.losses import MSE
from keras.optimizers import Adam
from ActorCriticFamily.common.ReplayBuffers import ReplayBufferNP
from ActorCriticFamily.common.NeuralNetworks import actor_network, critic_network

class Agent:
    def __init__(self, alpha, beta, tau, env, model_path,
                 gamma=0.99, update_actor_interval=2, cnt=None,
                 buffer_size=1_000_000, actor_architecture=None,
                 critic_architecture=None, batch_size=100, noise=0.1):

        input_dims = env.observation_space.shape
        actions_dims = env.action_space.shape
        input_action_concat = (input_dims[0] + actions_dims[0],)

        if cnt is None:
            self.cnt = 0
        else:
            self.cnt = cnt
        self.tau = tau
        self.gamma = gamma
        self.learn_step_cnt = 0
        self.model_path = model_path
        self.batch_size = batch_size
        self.n_actions = actions_dims[0]
        self.min_action = env.action_space.low[0]
        self.max_action = env.action_space.high[0]
        self.update_actor_iter = update_actor_interval
        self.memory = ReplayBufferNP(buffer_size, input_dims, actions_dims[0])

        self.actor = actor_network(layers=actor_architecture, input_dims=input_dims, actions_dims=actions_dims[0])
        self.critic_1 = critic_network(layers=critic_architecture, input_dims=input_action_concat)
        self.critic_2 = critic_network(layers=critic_architecture, input_dims=input_action_concat)
        self.target_actor = actor_network(layers=actor_architecture, input_dims=input_dims,
                                          actions_dims=actions_dims[0])
        self.target_critic_1 = critic_network(layers=critic_architecture, input_dims=input_action_concat)
        self.target_critic_2 = critic_network(layers=critic_architecture, input_dims=input_action_concat)

        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_actor.summary()
        self.noise = noise
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        # returns a batch size of 1, want a scalar array
        mu = self.actor(state)[0]
        mu_prime = mu + np.random.normal(scale=self.noise)

        mu_prime = tf.clip_by_value(mu_prime, self.min_action, self.max_action)

        return mu_prime

    def remember(self, state, action, reward, new_state, done):
        self.memory.store(state, action, reward, new_state, done)

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

    def save_models(self):
        self.cnt += 1
        print(f'Saving... #{self.cnt}')
        self.actor.save_weights(self.model_path + f"Actor_{self.cnt}", save_format="tf")
        self.critic_1.save_weights(self.model_path + f"Critic1_{self.cnt}", save_format="tf")
        self.critic_2.save_weights(self.model_path + f"Critic2_{self.cnt}", save_format="tf")
        self.target_actor.save_weights(self.model_path + f"TargetActor_{self.cnt}", save_format="tf")
        self.target_critic_1.save_weights(self.model_path + f"TargetCritic1_{self.cnt}", save_format="tf")
        self.target_critic_2.save_weights(self.model_path + f"TargetCritic2_{self.cnt}", save_format="tf")

    def load_models(self):
        print('... loading models ...')
        self.actor.load_weights(self.model_path + f"Actor_{self.cnt}")
        self.critic_1.load_weights(self.model_path + f"Critic1_{self.cnt}")
        self.critic_2.load_weights(self.model_path + f"Critic2_{self.cnt}")
        self.target_actor.load_weights(self.model_path + f"TargetActor_{self.cnt}")
        self.target_critic_1.load_weights(self.model_path + f"TargetCritic1_{self.cnt}")
        self.target_critic_2.load_weights(self.model_path + f"TargetCritic2_{self.cnt}")
