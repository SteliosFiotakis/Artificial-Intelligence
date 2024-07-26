"""
Conventions:
- For storing memories function name should be: "store()"
- For sampling memories function name should be: "sample()"
"""

import random
import numpy as np
from collections import deque

class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

class ReplayBufferNP:
    """
    A class for memory functionality with numpy implementation.
    """
    def __init__(self, max_size, state_dims, action_dims):
        """
        :param max_size: The maximum number of memories.
        :param state_dims: The shape of the state.
        :param action_dims: The shape of the action.
        """
        self.mem_cntr = 0
        self.mem_size = max_size
        self.state_memory = np.zeros((self.mem_size, *state_dims))
        self.new_state_memory = np.zeros((self.mem_size, *state_dims))
        self.action_memory = np.zeros((self.mem_size, action_dims))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)

    def store(self, state, action, reward, next_state, done):
        """
        A function that stores the state, action, reward, next state, termination
        given as arguments.
        :param state: The state to be stored,
        :param action: The action to be stored,
        :param reward: The reward to be stored,
        :param next_state: The next state to be stored,
        :param done: The termination condition to be stored.
        """
        index = self.mem_cntr % self.mem_size

        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample(self, number_of_memories):
        """
        A function that samples some memory, which the
        number is determined by the argument given.
        :param number_of_memories: The number of memories to be given.
        :return: 5 arrays of length "number_of_memories" (state, action, reward, next state, termination).
        """
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, number_of_memories, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

# ReplayBufferNP more efficient than ReplayBuffer
