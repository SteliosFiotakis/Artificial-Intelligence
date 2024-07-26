import random
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

xposm = 0
yposm = 1
directionm = 2
leftm = 3
infrontm = 4
rightm = 5


rows, cols = (25, 25)
arr = [[0 for i in range(cols)] for j in range(rows)]


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
        self.cinner_map = arr

    def update_map(self, state):
        self.cinner_map[state[yposm]][state[xposm]] = floor_id
        if state[directionm] == North_id:
            self.cinner_map[state[yposm]][state[xposm-1]] = state[leftm]
            self.cinner_map[state[yposm-1]][state[xposm]] = state[infrontm]
            self.cinner_map[state[yposm]][state[xposm+1]] = state[rightm]
        if state[directionm] == East_id:
            self.cinner_map[state[yposm-1]][state[xposm]] = state[leftm]
            self.cinner_map[state[yposm]][state[xposm+1]] = state[infrontm]
            self.cinner_map[state[yposm+1]][state[xposm]] = state[rightm]
        if state[directionm] == South_id:
            self.cinner_map[state[yposm]][state[xposm+1]] = state[leftm]
            self.cinner_map[state[yposm-1]][state[xposm]] = state[infrontm]
            self.cinner_map[state[yposm]][state[xposm-1]] = state[rightm]
        if state[directionm] == West_id:
            self.cinner_map[state[yposm+1]][state[xposm]] = state[leftm]
            self.cinner_map[state[yposm]][state[xposm-1]] = state[infrontm]
            self.cinner_map[state[yposm-1]][state[xposm]] = state[rightm]

    def print_map(self):
        for row in self.cinner_map:
            print(row)


env = GridWorld("Map3.png")
kostakis = Agent("Karagiorgakis")
modell = Model(kostakis)
env.print_map()
print()


kostakis.spawn(env, 1, 1, East_id)
# kostakis.set_target(10, 19)
env.print_map()


for _ in range(10):
    observation = kostakis.observe()
    modell.update_map(observation)
