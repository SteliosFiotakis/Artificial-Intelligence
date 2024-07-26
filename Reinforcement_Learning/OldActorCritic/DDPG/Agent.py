import numpy as np
import tensorflow as tf
from keras.optimizers import Adam
from ReplaySystems import ReplayBuffer, ReplayBufferNP
from NeuralNetworks import actor_network, critic_network

class Agent:
    def __init__(self, model_path=None, replay_buffer_s=10_000, actor_architecture=None,
                 critic_architecture=None, gamma=0.99, do_load=False, batch_size=32,
                 environment=None, tau=1e-2, alpha=1e-03, beta=2e-03, noise=0.1,
                 model_n=None):

        self.target_actor = actor_network(environment.observation_space.shape,
                                          environment.action_space.shape[0], actor_architecture)
        self.actor = actor_network(environment.observation_space.shape,
                                   environment.action_space.shape[0], actor_architecture)
        self.target_critic = critic_network(environment.observation_space.shape,
                                            environment.action_space.shape, critic_architecture)
        self.critic = critic_network(environment.observation_space.shape,
                                     environment.action_space.shape, critic_architecture)

        self.actor.compile(optimizer=Adam(learning_rate=alpha))
        self.critic.compile(optimizer=Adam(learning_rate=beta))
        self.target_actor.compile(optimizer=Adam(learning_rate=alpha))
        self.target_critic.compile(optimizer=Adam(learning_rate=beta))

        self.replay_buffer = ReplayBufferNP(replay_buffer_s, environment.observation_space.shape,
                                            environment.action_space.shape[0])
        self.model_path = model_path
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        self.actions_dims = environment.action_space.shape[0]
        self.max_action = environment.action_space.high[0]
        self.min_action = environment.action_space.low[0]
        self.noise = noise
        self.cnt = 0

        self.soft_update(tau=1)

        if do_load:
            self.load(model_n)

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def choose_action(self, state):
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        actions = self.actor(state)
        actions += tf.random.normal(shape=[self.actions_dims], mean=0.0, stddev=self.noise)
        actions = tf.clip_by_value(actions, self.min_action, self.max_action)

        return actions[0]

    def learn(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
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

    def soft_updatel(self, tau=None):
        if tau is None:
            tau = self.tau

        weights = []
        targets = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_actor.set_weights(weights)

        weights = []
        targets = self.target_critic.weights
        for i, weight in enumerate(self.critic.weights):
            weights.append(weight * tau + targets[i]*(1-tau))
        self.target_critic.set_weights(weights)

    def soft_update(self, tau=None):
        actor_weights = np.array(self.actor.get_weights(), dtype=object)
        target_actor_weights = np.array(self.target_actor.get_weights(), dtype=object)
        updated_weights = tau * target_actor_weights + (1 - tau) * actor_weights
        self.target_actor.set_weights(updated_weights)

        critic_weights = np.array(self.critic.get_weights(), dtype=object)
        target_critic_weights = np.array(self.target_critic.get_weights(), dtype=object)
        updated_weights = tau * target_critic_weights + (1 - tau) * critic_weights
        self.target_critic.set_weights(updated_weights)

    def save(self):
        # with open(self.model_path, "r+") as f:
        #     number = f.read()
        #     value_m = str(int(number) + 1)
        #     f.seek(0)
        #     f.truncate()
        #     f.write(value_m)
        self.cnt += 1
        print(f'Saving... #{self.cnt}')
        self.actor.save(self.model_path + f"Actor_{self.cnt}.h5")
        self.critic.save(self.model_path + f"Critic_{self.cnt}.h5")
        self.target_actor.save(self.model_path + f"TargetActor_{self.cnt}.h5")
        self.target_critic.save(self.model_path + f"TargetCritic_{self.cnt}.h5")

    def load(self, value):
        # with open("./Saves/WholeModels/Indexer.txt", "r") as f:
        #     value = f.read()
        self.actor = tf.keras.models.load_model(self.model_path + f"Actor_{value}.h5")
        self.critic = tf.keras.models.load_model(self.model_path + f"Critic_{value}.h5")
        self.target_actor = tf.keras.models.load_model(self.model_path + f"TargetActor_{value}.h5")
        self.target_critic = tf.keras.models.load_model(self.model_path + f"TargetCritic_{value}.h5")
