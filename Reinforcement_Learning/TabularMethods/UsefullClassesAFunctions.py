# Class GridWorld is an environment constructor for GridWorld cases.
# When creating an object, it takes as parameter an image of number maze/map, and it converts to number 2D python list.
# The colors of the image must be becarefully chosen, as different colors represent different things.
# Coordinates are inverted (y, x). First element indicates the row (y) and the second one the column (x).
# Coordinates are represent as two parameters instead of one. They are not tuples but ys and xs.
# TODO: Change python lists to numpy arrays
# TODO: Create backups for PC and Python Scripts


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
        self.termination_reward = 0
        self.floor_reward = -1
        self.wall_reward = -10

        # Positions
        self.starting_pos = None
        self.target_pos = None
        self.agent_pos = None

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
        self.starting_pos = (row, column)

    def target(self, row, column):
        self.target_pos = (row, column)

    def spawn(self):
        self.agent_pos = self.starting_pos

    def despawn(self):
        self.agent_pos = None

    def observe(self):
        return self.agent_pos

    def terminal_check(self, pos):
        if pos == self.target_pos:
            return True
        return False

    def step(self, action):
        if action == 0:
            y_cor, x_cor = self.agent_pos[0]-1, self.agent_pos[1]
        elif action == 1:
            y_cor, x_cor = self.agent_pos[0], self.agent_pos[1]+1
        elif action == 2:
            y_cor, x_cor = self.agent_pos[0]+1, self.agent_pos[1]
        elif action == 3:
            y_cor, x_cor = self.agent_pos[0], self.agent_pos[1]-1
        if self.cmap[y_cor][x_cor] == self.floor_id:
            self.agent_pos = (y_cor, x_cor)
            if (y_cor, x_cor) == self.target_pos:
                return self.termination_reward
            return self.floor_reward
        return self.wall_reward

    def create_transition_matrix(self):
        transition_dict = {}
        states_list = self.get_states()
        self.floor_reward = -1
        self.wall_reward = -1
        self.termination_reward = 0

        for action in range(4):
            for state in states_list:
                next_state, reward = self.check_move(state, action)
                transition_dict[state, action] = next_state, reward

        return transition_dict

    def check_move(self, state, action):
        if action == 0:
            y_cor, x_cor = state[0]-1, state[1]
        elif action == 1:
            y_cor, x_cor = state[0], state[1]+1
        elif action == 2:
            y_cor, x_cor = state[0]+1, state[1]
        elif action == 3:
            y_cor, x_cor = state[0], state[1]-1
        if self.terminal_check((y_cor, x_cor)):
            return (y_cor, x_cor), self.termination_reward
        elif self.cmap[y_cor][x_cor] == self.wall_id:
            return state, self.wall_reward
        elif self.cmap[y_cor][x_cor] == self.floor_id:
            return (y_cor, x_cor), self.floor_reward


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
                array[row_idx, value_idx] = [0, 0, 255]
            elif value == 4:
                array[row_idx, value_idx] = [255, 0, 0]

    img = Image.fromarray(array)

    plt.title("GridWorld")
    # plt.xlabel("X")
    # plt.ylabel("Y")

    plt.imshow(img)
    plt.pause(0.1)
