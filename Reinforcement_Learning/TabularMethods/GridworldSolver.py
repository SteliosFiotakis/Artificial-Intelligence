# Documentation
# Map creation: https://www.pixilart.com
# Class GridWorld: Environment constructor. It maps images to numerical arrays representing the map configuration.
#
# -----------------
#
# Array values reference:
# -1: An Agent
# 0: Free moving
# 1: Wall
#
# -----------------
#
# Directions mapping:
# 0: North_id / Up
# 1: East_id / TurnRight_id
# 2: South_id / Down
# 3: West_id / TurnLeft_id
#
# -----------------
#
# Actions:
# 0: Move forward
# 1: Move backward
# 2: Turn left
# 3: Turn TurnRight_id
#
# -----------------
#
#
#
#
#
#
#
#
# variables ending with "_id" refers to global mapping variables
#
#
#

import random
import keyboard
from PIL import Image
import numpy as np

wall_id = 1
floor_id = 0
agent_id = 2
dummy_id = 5
North_id = 0
East_id = 1
South_id = 2
West_id = 3
Forward_id = 0
Backward_id = 1
TurnLeft_id = 2
TurnRight_id = 3

find = {'xpos': 0, 'ypos': 1, 'direction': 2, 'left': 3, 'infront': 4, 'right': 5}

xposm = 0
yposm = 1
directionm = 2
leftm = 3
infrontm = 4
rightm = 5


class GridWorld:
    def __init__(self, map_img):
        self.cmap = GridWorld.map_maker(map_img)

    @staticmethod
    def map_maker(image):
        row_list = []
        my_list = []

        img = Image.open(image)
        data = np.asarray(img)

        for arr_row in data:
            for pixel in arr_row:
                if max(pixel):
                    row_list.append(wall_id)
                else:
                    row_list.append(floor_id)
            my_list.append(row_list.copy())
            row_list.clear()

        return my_list

    def print_map(self):
        for row in self.cmap:
            print(row)

    # Gives info of the surrounding tiles
    # def observe(self):
    #     if self.cdirection == 0:
    #         return self.cx, self.cy, self.cy-1, self.cx-1, self.cx+1    # x-pos, y-pos, ahead, left, right
    #     elif self.cdirection == 1:
    #         return self.cx, self.cy, self.cx+1, self.cy-1, self.cy+1    # x-pos, y-pos, ahead, left, right
    #     elif self.cdirection == 2:
    #         return self.cx, self.cy, self.cy+1, self.cx+1, self.cx-1    # x-pos, y-pos, ahead, left, right
    #     elif self.cdirection == 3:
    #         return self.cx, self.cy, self.cx-1, self.cy+1, self.cy-1    # x-pos, y-pos, ahead, left, right
    #     else:
    #         print("Error: class variable 'cdirection', was taken wrong value.")
    #         return None
    # def do_action(self, action):
    #     self.caction = action
    #     if self.caction == 1:
    #         if not self.cmap[self.cy][self.cx + 1]:
    #             self.cx += 1
    #             self.creward = -1
    #             if self.cx == self.ctarget_x and self.cy == self.ctarget_y:
    #                 self.cdone = True
    #                 self.creward = 1001
    #         else:
    #             self.creward = -1000
    #         return self.creward, self.cdone
    #     if self.caction == 3:
    #         if not self.cmap[self.cy][self.cx - 1]:
    #             self.cx -= 1
    #             self.creward = -1
    #             if self.cx == self.ctarget_x and self.cy == self.ctarget_y:
    #                 self.cdone = True
    #                 self.creward = 1001
    #         else:
    #             self.creward = -1000
    #         return self.creward, self.cdone
    #     if self.caction == 2:
    #         if not self.cmap[self.cy + 1][self.cx]:
    #             self.cy += 1
    #             self.creward = -1
    #             if self.cx == self.ctarget_x and self.cy == self.ctarget_y:
    #                 self.cdone = True
    #                 self.creward = 1001
    #         else:
    #             self.creward = -1000
    #         return self.creward, self.cdone
    #     if not self.caction:
    #         if not self.cmap[self.cy - 1][self.cx]:
    #             self.cy -= 1
    #             self.creward = -1
    #             if self.cx == self.ctarget_x and self.cy == self.ctarget_y:
    #                 self.cdone = True
    #                 self.creward = 1001
    #         else:
    #             self.creward = -1000
    #         return self.creward, self.cdonesss

    def physics(self):
        pass

    def change(self, agent, agent_xpos, agent_ypos, agent_direction, action):
        if agent_direction == North_id:
            if action == Forward_id:
                if not self.pos_check(agent_xpos, agent_ypos-1):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos-1][agent_xpos] = agent_id
                    agent.cy -= 1
            if action == Backward_id:
                if not self.pos_check(agent_xpos, agent_ypos+1):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos+1][agent_xpos] = agent_id
                    agent.cy += 1
        if agent_direction == East_id:
            if action == Forward_id:
                if not self.pos_check(agent_xpos+1, agent_ypos):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos][agent_xpos+1] = agent_id
                    agent.cx += 1
            if action == Backward_id:
                if not self.pos_check(agent_xpos-1, agent_ypos):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos][agent_xpos-1] = agent_id
                    agent.cx -= 1
        if agent_direction == South_id:
            if action == Forward_id:
                if not self.pos_check(agent_xpos, agent_ypos+1):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos+1][agent_xpos] = agent_id
                    agent.cy += 1
            if action == Backward_id:
                if not self.pos_check(agent_xpos, agent_ypos-1):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos-1][agent_xpos] = agent_id
                    agent.cy -= 1
        if agent_direction == West_id:
            if action == Forward_id:
                if not self.pos_check(agent_xpos-1, agent_ypos):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos][agent_xpos-1] = agent_id
                    agent.cx -= 1
            if action == Backward_id:
                if not self.pos_check(agent_xpos+1, agent_ypos):
                    self.cmap[agent_ypos][agent_xpos] = floor_id
                    self.cmap[agent_ypos][agent_xpos+1] = agent_id
                    agent.cx += 1

    def pos_check(self, xpos, ypos):
        return self.cmap[ypos][xpos]


class Agent:
    def __init__(self, name):
        self.name = name
        self.cx = None
        self.cy = None
        self.cx_target = None
        self.cy_target = None
        self.cdirection = None
        self.cenvironment = None
        self.cmemories = list()

    # Gives info of the surrounding tiles
    def observe(self):
        if self.cdirection == North_id:
            return (self.cx, self.cy, self.cdirection,
                    self.cenvironment.cmap[self.cy][self.cx-1],
                    self.cenvironment.cmap[self.cy-1][self.cx],
                    self.cenvironment.cmap[self.cy][self.cx+1])
        elif self.cdirection == East_id:
            return (self.cx, self.cy, self.cdirection,
                    self.cenvironment.cmap[self.cy-1][self.cx],
                    self.cenvironment.cmap[self.cy][self.cx+1],
                    self.cenvironment.cmap[self.cy+1][self.cx])
        elif self.cdirection == South_id:
            return (self.cx, self.cy, self.cdirection,
                    self.cenvironment.cmap[self.cy][self.cx+1],
                    self.cenvironment.cmap[self.cy+1][self.cx],
                    self.cenvironment.cmap[self.cy][self.cx-1])
        elif self.cdirection == West_id:
            return (self.cx, self.cy, self.cdirection,
                    self.cenvironment.cmap[self.cy+1][self.cx],
                    self.cenvironment.cmap[self.cy][self.cx-1],
                    self.cenvironment.cmap[self.cy-1][self.cx])
        else:
            print("Error: class variable 'cdirection', was taken wrong value.")
            return None

    def take_action(self, action):
        if action == TurnLeft_id:
            self.cdirection = self.cdirection-1 if self.cdirection != North_id else West_id
        elif action == TurnRight_id:
            self.cdirection = self.cdirection+1 if self.cdirection != West_id else North_id
        elif action == Forward_id or action == Backward_id:
            self.cenvironment.change(self, self.cx, self.cy, self.cdirection, action)
        else:
            print("Non valid action given")
            return None

    def spawn(self, environment, xpos, ypos, direction=North_id):
        self.cenvironment = environment
        if environment.pos_check(xpos, ypos) == floor_id:
            environment.cmap[ypos][xpos] = agent_id
            self.cx = xpos
            self.cy = ypos
            self.cdirection = direction
        else:
            print("Location given is not valid")
            return None

    def looking_direction(self):
        print(self.cdirection)

    def add_to_memory(self, astate, aaction, anew_state):
        if (astate, aaction, anew_state) not in self.cmemories:
            self.cmemories.append((astate, aaction, anew_state))

    def set_target(self, xpos, ypos):
        self.cx_target = xpos
        self.cy_target = ypos

    def save_memories(self):
        pass

    def load_memories(self):
        pass


class Model:
    def __init__(self, agent):
        self.cagent = agent
        self.cinner_map = self.create_array(25, 25)

    @staticmethod
    def create_array(height, width):
        rows, cols = (height, width)
        arr = [[dummy_id for _ in range(cols)] for _ in range(rows)]
        return arr

    def update_map(self, state):
        self.cinner_map[state[yposm]][state[xposm]] = floor_id
        if state[directionm] == North_id:
            self.cinner_map[state[yposm]][state[xposm]-1] = state[leftm]
            self.cinner_map[state[yposm]-1][state[xposm]] = state[infrontm]
            self.cinner_map[state[yposm]][state[xposm]+1] = state[rightm]
        if state[directionm] == East_id:
            self.cinner_map[state[yposm]-1][state[xposm]] = state[leftm]
            self.cinner_map[state[yposm]][state[xposm]+1] = state[infrontm]
            self.cinner_map[state[yposm]+1][state[xposm]] = state[rightm]
        if state[directionm] == South_id:
            self.cinner_map[state[yposm]][state[xposm]+1] = state[leftm]
            self.cinner_map[state[yposm]+1][state[xposm]] = state[infrontm]
            self.cinner_map[state[yposm]][state[xposm]-1] = state[rightm]
        if state[directionm] == West_id:
            self.cinner_map[state[yposm]+1][state[xposm]] = state[leftm]
            self.cinner_map[state[yposm]][state[xposm]-1] = state[infrontm]
            self.cinner_map[state[yposm]-1][state[xposm]] = state[rightm]

    def print_map(self):
        for row in self.cinner_map:
            print(row)

    def save_model(self, path):
        with open(path, 'w') as f:
            for row in self.cinner_map:
                for value in row:
                    f.write(str(value) + ',')
                f.write('\n')

    def load_model(self, path):
        temp_sub_list = list()
        temp_list = list()
        with open(path, 'r') as f:
            for line in f:
                my_str = line.split(',')
                for value in my_str[:-1]:
                    temp_sub_list.append(int(value))
                temp_list.append(temp_sub_list.copy())
                temp_sub_list.clear()
        self.cinner_map = temp_list


env = GridWorld("Map3.png")
kostakis = Agent("Karagiorgakis")
kostasmodel = Model(kostakis)


kostakis.spawn(env, 1, 1, East_id)
kostakis.set_target(10, 19)
kostasmodel.update_map(kostakis.observe())
kostasmodel.print_map()


'''
for _ in range(10):
    action_taken = random.randint(0, 3)
    state = kostakis.observe()
    kostakis.take_action(action_taken)
    kostasmodel.update_map(state)
    print(state)
    print(action_taken)
    kostasmodel.print_map()
    print()


while True:
    if keyboard.read_key() == "w":
        kostakis.take_action(0)
        print()
        state = kostakis.observe()
        kostasmodel.update_map(state)
        kostasmodel.print_map()
    if keyboard.read_key() == "number":
        kostakis.take_action(2)
        print()
        state = kostakis.observe()
        kostasmodel.update_map(state)
        kostasmodel.print_map()
    if keyboard.read_key() == "s":
        kostakis.take_action(1)
        print()
        state = kostakis.observe()
        kostasmodel.update_map(state)
        kostasmodel.print_map()
    if keyboard.read_key() == "d":
        kostakis.take_action(3)
        print()
        state = kostakis.observe()
        kostasmodel.update_map(state)
        kostasmodel.print_map()
    if keyboard.read_key() == "p":
        break

kostasmodel.save_model("my_map.txt")

#     new_observation = kostakis.observe()
#     muscle_memory_table.append((state, action_taken, new_observation))
#     print()
#     print(kostakis.cdirection)
#     env.print_map()
#
# print(muscle_memory_table)


# kostasmodel.update_map(state)
# print()
# kostasmodel.print_map()
# env.print_map()
# print(state)

# print()
# kostasmodel.print_map()


# print(kostakis.observe())
# env1.print_map()
#
# kostakis.take_action(0)
# print(kostakis.observe())
# env1.print_map()
#
# kostakis.take_action(0)
# print(kostakis.observe())
# env1.print_map()
#
# kostakis.take_action(3)
# print(kostakis.observe())
# env1.print_map()
#
# Giorgos = Agent("Agent47")
#
# env1.print_map()
#
# Giorgos.spawn(env1, 1, 1, 1)
#
# env1.print_map()
#
# Giorgos.take_action(0)
# env1.print_map()
#
# Giorgos.take_action(0)
# env1.print_map()
#
# while True:
#     if keyboard.read_key() == "w":
#         Giorgos.take_action(0)
#         env1.print_map()
#         Giorgos.looking_direction()
#     if keyboard.read_key() == "number":
#         Giorgos.take_action(2)
#         env1.print_map()
#         Giorgos.looking_direction()
#     if keyboard.read_key() == "s":
#         Giorgos.take_action(1)
#         env1.print_map()
#         Giorgos.looking_direction()
#     if keyboard.read_key() == "d":
#         Giorgos.take_action(3)
#         env1.print_map()
#         Giorgos.looking_direction()
#     if keyboard.read_key() == "p":
#         break
#
#
# env.spawn((15, 17))
# env.target((10, 19))
#
#
# # Printing map
# for row in env.cmap:
#     print(row)
#
#
# # Map states to q-table
# cnt = 0
# my_dict = {}
# for row in range(19):
#     for column in range(21):
#         my_dict[(row, column)] = cnt
#         cnt += 1
# print(my_dict)
#
#
# # Q learning Algorithm
#
# q_table = np.zeros((len(my_dict), 4))
# # for elem in q_table:
# #     print(elem)
#
# total_training_steps = 1000
# steps = 200
#
# alpha = 1
# gamma = 1
#
#
# epsilon = 1
# epsilon_decay = 0.0001
#
# for episode in range(total_training_steps):
#     state = env.observe()
#     episode_rewards = 0
#
#     for step in range(steps):
#         # Choosing action
#         if random.random() > epsilon:
#             reward, termination = env.do_action(random.randint(0, 3))
#         else:
#             reward, termination = env.do_action(np.argmax(q_table[my_dict[state]]))
#
#         next_state = env.observe()
#
#         # Updating q_table
#         q_table[my_dict[state]][env.caction] = q_table[my_dict[state]][env.caction] + alpha * \
#             (reward + gamma * max(q_table[my_dict[next_state]]) - q_table[my_dict[state]][env.caction])
#
#         state = next_state
#
#         if termination:
#             break
#
#     # Epsilon decay
#     epsilon += epsilon_decay
#
#     env.spawn((15, 17))
#     env.cdone = False
#
#
# for key, value in my_dict.items():
#     print(key, q_table[value])
#
#
# state = env.observe()
# print(state)
# env.cdone = False
# episode_rewards = 0
#
# for step in range(steps):
#     reward, termination = env.do_action(np.argmax(q_table[my_dict[state]]))
#     print(env.caction)
#
#     state = env.observe()
#     print(state)
#
#     if termination:
#         break

'''
