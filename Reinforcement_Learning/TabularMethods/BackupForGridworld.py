# Class GridWorld is an environment constructor for GridWorld cases.
# When creating an object, it takes as parameter an image of number maze/map, and it converts to number 2D python list.
# The colors of the image must be becarefully chosen, as different colors represent different things.
# Coordinates are inverted (y, x). First element indicates the row (y) and the second one the column (x).
# TODO: Change python lists to numpy arrays


import copy
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


class GridWorld:
    floor_id = 0
    wall_id = 1
    starting_id = 2
    agent_id = 3
    target_id = 4

    def __init__(self, map_img):
        # Maps
        self.cmap = GridWorld.map_maker(map_img)
        self.representation_map = copy.deepcopy(self.cmap)

        # Rewards
        self.termination_reward = None
        self.floor_reward = None
        self.wall_reward = None

        # Positions
        self.starting_pos = None
        self.target_pos = None
        self.agent_pos = None

        # Existing Starting/Target Positions
        self.start_exist = False
        self.target_exist = False

    @staticmethod
    def map_maker(image):
        row_list = []
        my_list = []

        img = Image.open(image)
        data = np.asarray(img)

        for arr_row in data:
            for pixel in arr_row:
                if max(pixel):
                    row_list.append(GridWorld.wall_id)
                else:
                    row_list.append(GridWorld.floor_id)
            my_list.append(row_list.copy())
            row_list.clear()

        return my_list

    def print_map(self):
        print('\nMap:')
        for row in self.representation_map:
            print(row)
        print()

    def get_states(self):
        list_of_states = list()
        for row_idx, row in enumerate(self.cmap):
            for column_idx, column in enumerate(row):
                if column == self.floor_id:
                    list_of_states.append((row_idx, column_idx))
        return list_of_states

    def starting(self, row, column):
        # if self.start_exist:
        #     self.representation_map[self.starting_pos[0]][self.starting_pos[1]] = self.floor_id
        self.starting_pos = (row, column)
        # self.representation_map[self.starting_pos[0]][self.starting_pos[1]] = self.starting_id
        self.start_exist = True

    def target(self, row, column):
        # if self.target_exist:
        #     self.representation_map[self.target_pos[0]][self.target_pos[1]] = self.floor_id
        self.target_pos = (row, column)
        # self.representation_map[self.target_pos[0]][self.target_pos[1]] = self.target_id
        self.target_exist = True

    def spawn(self):
        self.agent_pos = self.starting_pos.copy()
        # self.representation_map[self.starting_pos[0]][self.starting_pos[1]] = self.agent_id

    def despawn(self):
        pass

    # def reset(self):
    #     # self.representation_map = copy.deepcopy(self.cmap)
    #     # self.representation_map[self.starting_pos[0]][self.starting_pos[1]] = self.starting_id
    #     # self.representation_map[self.target_pos[0]][self.target_pos[1]] = self.target_id
    #     pass

    def observe(self):
        return self.agent_pos

    def terminal_check(self, pos):
        if pos == self.target_pos:
            return True
        else:
            return False

    def step(self, action):     # TODO CREATE EPISODIC ENV CHANGES
        if action == 0:
            coordinates = (self.agent_pos[0]-1, self.agent_pos[1])
        elif action == 1:
            coordinates = (self.agent_pos[0], self.agent_pos[1]+1)
        elif action == 2:
            coordinates = (self.agent_pos[0]+1, self.agent_pos[1])
        elif action == 3:
            coordinates = (self.agent_pos[0], self.agent_pos[1]-1)
        if self.representation_map[coordinates[0]][coordinates[1]] == self.floor_id:
            self.agent_pos = coordinates

    def create_transition_matrix(self):
        transtion_dict = {}
        states_list = self.get_states()
        self.floor_reward = -1
        self.wall_reward = -1   # - (len(states_list) ** 2)
        self.termination_reward = 0

        for action in range(4):
            for state in states_list:
                next_state, reward = self.check_move(state, action)
                transtion_dict[state, action] = next_state, reward

        return transtion_dict

    def check_move(self, state, action):
        if action == 0:
            if self.terminal_check((state[0]-1, state[1])):
                return (state[0]-1, state[1]), self.termination_reward
            elif self.representation_map[state[0] - 1][state[1]] == self.wall_id:
                return state, self.wall_reward
            else:
                return (state[0]-1, state[1]), self.floor_reward
        elif action == 1:
            if self.terminal_check((state[0], state[1]+1)):
                return (state[0], state[1]+1), self.termination_reward
            elif self.representation_map[state[0]][state[1] + 1] == self.wall_id:
                return state, self.wall_reward
            else:
                return (state[0], state[1]+1), self.floor_reward
        elif action == 2:
            if self.terminal_check((state[0]+1, state[1])):
                return (state[0]+1, state[1]), self.termination_reward
            elif self.representation_map[state[0] + 1][state[1]] == self.wall_id:
                return state, self.wall_reward
            else:
                return (state[0]+1, state[1]), self.floor_reward
        elif action == 3:
            if self.terminal_check((state[0], state[1]-1)):
                return (state[0], state[1]-1), self.termination_reward
            elif self.representation_map[state[0]][state[1] - 1] == self.wall_id:
                return state, self.wall_reward
            else:
                return (state[0], state[1]-1), self.floor_reward


def visualize(a_array):
    width = len(a_array[0])
    height = len(a_array)
    array = np.zeros([height, width, 3], dtype=np.uint8)

    for row_idx, row in enumerate(a_array):
        for value_idx, value in enumerate(row):
            if value == 0:
                array[row_idx, value_idx] = [255, 255, 255]
            elif value == 1:
                array[row_idx, value_idx] = [0, 0, 0]
            elif value == 2:
                array[row_idx, value_idx] = [0, 255, 0]
            elif value == 3:
                array[row_idx, value_idx] = [255, 0, 0]

    img = Image.fromarray(array)

    plt.title("GridWorld")
    # plt.xlabel("X")
    # plt.ylabel("Y")

    plt.imshow(img)
    plt.pause(0.1)
