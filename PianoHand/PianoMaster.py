from scipy.spatial.transform import Rotation as R
from mujoco.glfw import glfw
import mujoco as mj
import pygame as pg
import numpy as np
import asyncio
import os

pg.init()
pg.mixer.init()


def synth(frequency, duration=1.5, sampling_rate=44_100):
    frames = int(duration * sampling_rate)
    arr = np.cos(2 * np.pi * frequency * np.linspace(0, duration, frames))
    arr = arr + np.cos(4 * np.pi * frequency * np.linspace(0, duration, frames))
    arr = arr - np.cos(6 * np.pi * frequency * np.linspace(0, duration, frames))
    arr = np.clip(arr*10, -1, 1)    # squarish waves
    # arr = np.cumsum(np.clip(arr*10, -1, 1))   # triangularish waves pt1
    # arr = arr+np.sin(2*np.pi*frequency*np.linspace(0,duration, frames))   # triangularish waves pt1
    # arr = arr / max(np.abs(arr))  # triangularish waves pt1
    sound = np.asarray([32_767 * arr, 32_767 * arr]).T.astype(np.int16)
    sound = pg.sndarray.make_sound(sound.copy())

    return sound


with open("notesdict.txt", "r") as file:
    frequencies_dict = eval(file.readline())


xml_path = 'assets/hand/reach.xml'  # xml file (assumes this is in the same folder as this file)
simend = 1_000  # simulation time
print_camera_config = 0  # set to 1 to print camera config this is useful for initializing view of the model

# For callback functions
button_left = False
button_middle = False
button_right = False
lastx = 0
lasty = 0
height = 2

# initial_pos = np.asarray([-4.00000000e-01, -3.00000000e-01,  3.00000000e-01,  4.00000000e-01,
#   6.00000000e-01,  2.00000000e-01,  5.00000000e-01,  4.00000000e-01,
#   2.00000000e-01,  6.00000000e-01,  5.00000000e-01,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.00000000e-01,
#   3.00000000e-01,  2.00000000e-01,  2.77555756e-17, -2.00000000e-01])
#
# g_sharp_pos = np.asarray([-4.00000000e-01, -3.00000000e-01,  3.00000000e-01,  4.00000000e-01,
#   6.00000000e-01,  2.00000000e-01,  5.00000000e-01,  4.00000000e-01,
#   2.00000000e-01,  6.00000000e-01,  5.00000000e-01,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.00000000e-01,
#   4.00000000e-01,  2.00000000e-01,  2.77555756e-17, -2.00000000e-01])
#
# c_sharp_pos = np.asarray([-4.00000000e-01, -3.00000000e-01,  3.00000000e-01,  6.00000000e-01,
#   6.00000000e-01,  2.00000000e-01,  5.00000000e-01,  4.00000000e-01,
#   2.00000000e-01,  6.00000000e-01,  5.00000000e-01,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.00000000e-01,
#   3.00000000e-01,  2.00000000e-01,  2.77555756e-17, -2.00000000e-01])
#
# e_pos = np.asarray([-4.00000000e-01, -3.00000000e-01,  3.00000000e-01,  4.00000000e-01,
#   6.00000000e-01,  2.00000000e-01,  5.00000000e-01,  4.00000000e-01,
#   2.00000000e-01,  8.00000000e-01,  5.00000000e-01,  0.00000000e+00,
#   0.00000000e+00,  0.00000000e+00,  0.00000000e+00, -2.00000000e-01,
#   3.00000000e-01,  2.00000000e-01,  2.77555756e-17, -2.00000000e-01])


# main_init = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
#                         4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
#                         4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
#                         0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
#                         2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
#                         -1.00000000e-01, 1.60000000e-01, -5.00000000e-02])
# main_a_sharp = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
#                            4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
#                            4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
#                            0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
#                            4.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
#                            -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
# main_d = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 6.00000000e-01,
#                      4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
#                      4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
#                      0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
#                      2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
#                      -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
# main_g = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
#                      4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
#                      4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
#                      0.00000000e+00, 2.00000000e-01, 5.00000000e-01, 0.00000000e+00,
#                      2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
#                      -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
# main_f = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
#                      4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
#                      4.00000000e-01, 5.00000000e-01, 6.00000000e-01, 3.00000000e-01,
#                      0.00000000e+00, 1.00000000e-01, 5.00000000e-01, 0.00000000e+00,
#                      2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
#                      -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
#
# gaasharpd_pre_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
#                                  6.00000000e-01, 4.00000000e-01, 5.00000000e-01, 2.00000000e-01,
#                                  2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                                  -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                                  4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                                  -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# gaasharpd_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
#                              6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
#                              2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                              -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                              4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                              -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# gaasharpd_g = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
#                           6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
#                           2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           7.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                           -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# gaasharpd_a = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 7.00000000e-01,
#                           6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
#                           2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                           -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# gaasharpd_a_sharp = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
#                                 6.00000000e-01, 4.00000000e-01, 9.00000000e-01, 2.00000000e-01,
#                                 2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                                 -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                                 4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                                 -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# gaasharpd_d = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
#                           6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
#                           2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           -1.00000000e-01, 6.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                           4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
#                           -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
#
# acdef_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
#                          6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
#                          2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                          -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                          3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                          -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# acdef_a = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
#                       6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
#                       2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                       -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                       5.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                       -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# acdef_c = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 7.00000000e-01,
#                       6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
#                       2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                       -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                       3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                       -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# acdef_d = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
#                       6.00000000e-01, 2.77555756e-17, 7.00000000e-01, 6.00000000e-01,
#                       2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                       -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                       3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                       -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# acdef_e = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
#                       6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
#                       2.77555756e-17, 7.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                       -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                       3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                       -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
# acdef_f = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
#                       6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
#                       2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
#                       -1.00000000e-01, 6.00000000e-01, 6.00000000e-01, -2.00000000e-01,
#                       3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
#                       -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])


def quat2euler(quat_mujoco):
    # mujocoy quat is constant,x,y,z,
    # scipy quaut is x,y,z,constant
    quat_scipy = np.array([quat_mujoco[3], quat_mujoco[0], quat_mujoco[1], quat_mujoco[2]])

    r = R.from_quat(quat_scipy)
    euler = r.as_euler('xyz', degrees=True)

    return euler


def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    data.ctrl[46:] = -0.5


def controller(model, data):
    # put the controller here. This function is called inside the simulation.
    # print(data.site_xpos)
    # mujoco.MjData.xpos
    # mujoco.MjData.site_xpos
    # global wait
    # global counter
    # global time_list
    # if wait:
    #     time_passed = data.time - counter
    #     if time_passed < time_list[0]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[1]:
    #         data.ctrl[:23] = main_a_sharp
    #     elif time_passed < time_list[2]:
    #         data.ctrl[:23] = main_d
    #     elif time_passed < time_list[3]:
    #         data.ctrl[:23] = main_g
    #
    #     elif time_passed < time_list[4]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[5]:
    #         data.ctrl[:23] = main_g
    #     elif time_passed < time_list[6]:
    #         data.ctrl[:23] = main_f
    #     elif time_passed < time_list[7]:
    #         data.ctrl[:23] = main_d
    #
    #     elif time_passed < time_list[8]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[9]:
    #         data.ctrl[:23] = main_a_sharp
    #     elif time_passed < time_list[10]:
    #         data.ctrl[:23] = main_d
    #     elif time_passed < time_list[11]:
    #         data.ctrl[:23] = main_g
    #
    #     elif time_passed < time_list[12]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[13]:
    #         data.ctrl[:23] = main_g
    #     elif time_passed < time_list[14]:
    #         data.ctrl[:23] = main_f
    #     elif time_passed < time_list[15]:
    #         data.ctrl[:23] = main_d
    #
    #     elif time_passed < time_list[16]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[17]:
    #         data.ctrl[:23] = main_a_sharp
    #     elif time_passed < time_list[18]:
    #         data.ctrl[:23] = main_d
    #     elif time_passed < time_list[19]:
    #         data.ctrl[:23] = main_g
    #
    #     elif time_passed < time_list[20]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[21]:
    #         data.ctrl[:23] = main_g
    #     elif time_passed < time_list[22]:
    #         data.ctrl[:23] = main_f
    #     elif time_passed < time_list[23]:
    #         data.ctrl[:23] = main_d
    #
    #     elif time_passed < time_list[24]:
    #         data.ctrl[:23] = main_init
    #     elif time_passed < time_list[25]:
    #         data.ctrl[:23] = main_a_sharp
    #     elif time_passed < time_list[26]:
    #         data.ctrl[:23] = main_d
    #     elif time_passed < time_list[27]:
    #         data.ctrl[:23] = main_g
    #
    #     elif time_passed < time_list[28]:
    #         data.ctrl[:23] = gaasharpd_pre_init
    #     elif time_passed < time_list[29]:
    #         data.ctrl[:23] = gaasharpd_init
    #     elif time_passed < time_list[30]:
    #         data.ctrl[:23] = gaasharpd_g
    #     elif time_passed < time_list[31]:
    #         data.ctrl[:23] = gaasharpd_a
    #     elif time_passed < time_list[32]:
    #         data.ctrl[:23] = gaasharpd_a_sharp
    #     elif time_passed < time_list[33]:
    #         data.ctrl[:23] = gaasharpd_d
    #
    #     elif time_passed < time_list[34]:
    #         data.ctrl[:23] = acdef_init
    #     elif time_passed < time_list[35]:
    #         data.ctrl[:23] = acdef_e
    #     elif time_passed < time_list[36]:
    #         data.ctrl[:23] = acdef_f
    #     elif time_passed < time_list[37]:
    #         data.ctrl[:23] = acdef_d
    #     elif time_passed < time_list[38]:
    #         data.ctrl[:23] = acdef_e
    #     elif time_passed < time_list[39]:
    #         data.ctrl[:23] = acdef_c
    #     elif time_passed < time_list[40]:
    #         data.ctrl[:23] = acdef_d
    #     elif time_passed < time_list[41]:
    #         data.ctrl[:23] = acdef_a
    #     elif time_passed < time_list[42]:
    #         data.ctrl[:23] = acdef_init
    #     else:
    #         wait = False
    pass
    # runs every 0.002 seconds


hand_in_use = 'LEFT_HAND '
hand = 0 if hand_in_use else 23

kai = {
    'LEFT_HAND 44': 33,     # KEY_COMMA
    'LEFT_HAND 46': 37,     # KEY_PERIOD
    'LEFT_HAND 47': 37,     # KEY_SLASH
    'LEFT_HAND 48': 34,     # KEY_0
    'LEFT_HAND 49': 38,     # KEY_1
    'LEFT_HAND 50': 38,     # KEY_2
    'LEFT_HAND 51': 39,     # KEY_3
    'LEFT_HAND 52': 39,     # KEY_4
    'LEFT_HAND 53': 23,     # KEY_5
    'LEFT_HAND 54': 23,     # KEY_6
    'LEFT_HAND 55': 24,     # KEY_7
    'LEFT_HAND 56': 24,     # KEY_8
    'LEFT_HAND 57': 34,     # KEY_9
    'LEFT_HAND 59': 36,     # KEY_SEMICOLON
    'LEFT_HAND 65': 41,     # KEY_A
    'LEFT_HAND 66': 30,     # KEY_B
    'LEFT_HAND 67': 27,     # KEY_C
    'LEFT_HAND 68': 26,     # KEY_D
    'LEFT_HAND 69': 25,     # KEY_E
    'LEFT_HAND 70': 26,     # KEY_F
    'LEFT_HAND 71': 29,     # KEY_G
    'LEFT_HAND 72': 29,     # KEY_H
    'LEFT_HAND 73': 31,     # KEY_I
    'LEFT_HAND 74': 32,     # KEY_J
    'LEFT_HAND 75': 32,     # KEY_K
    'LEFT_HAND 76': 36,     # KEY_L
    'LEFT_HAND 77': 33,     # KEY_M
    'LEFT_HAND 78': 30,     # KEY_N
    'LEFT_HAND 79': 35,     # KEY_O
    'LEFT_HAND 80': 35,     # KEY_P
    'LEFT_HAND 81': 40,     # KEY_Q
    'LEFT_HAND 82': 25,     # KEY_R
    'LEFT_HAND 83': 41,     # KEY_S
    'LEFT_HAND 84': 28,     # KEY_T
    'LEFT_HAND 85': 31,     # KEY_U
    'LEFT_HAND 86': 27,     # KEY_V
    'LEFT_HAND 87': 40,     # KEY_W
    'LEFT_HAND 88': 42,     # KEY_X
    'LEFT_HAND 89': 28,     # KEY_Y
    'LEFT_HAND 90': 42,     # KEY_Z
    'LEFT_HAND 262': 43,    # KEY_RIGHT
    'LEFT_HAND 263': 43,    # KEY_LEFT
    'LEFT_HAND 264': 44,    # KEY_DOWN
    'LEFT_HAND 265': 44,    # KEY_UP
    'LEFT_HAND 266': 45,    # KEY_PAGE_UP
    'LEFT_HAND 267': 45,    # KEY_PAGE_DOWN

    'RIGHT_HAND 44': 10,    # KEY_COMMA
    'RIGHT_HAND 46': 14,    # KEY_PERIOD
    'RIGHT_HAND 47': 14,    # KEY_SLASH
    'RIGHT_HAND 48': 11,    # KEY_0
    'RIGHT_HAND 49': 15,    # KEY_1
    'RIGHT_HAND 50': 15,    # KEY_2
    'RIGHT_HAND 51': 16,    # KEY_3
    'RIGHT_HAND 52': 16,    # KEY_4
    'RIGHT_HAND 53': 0,     # KEY_5
    'RIGHT_HAND 54': 0,     # KEY_6
    'RIGHT_HAND 55': 1,     # KEY_7
    'RIGHT_HAND 56': 1,     # KEY_8
    'RIGHT_HAND 57': 11,    # KEY_9
    'RIGHT_HAND 59': 13,    # KEY_SEMICOLON
    'RIGHT_HAND 65': 18,    # KEY_A
    'RIGHT_HAND 66': 7,     # KEY_B
    'RIGHT_HAND 67': 4,     # KEY_C
    'RIGHT_HAND 68': 3,     # KEY_D
    'RIGHT_HAND 69': 2,     # KEY_E
    'RIGHT_HAND 70': 3,     # KEY_F
    'RIGHT_HAND 71': 6,     # KEY_G
    'RIGHT_HAND 72': 6,     # KEY_H
    'RIGHT_HAND 73': 8,     # KEY_I
    'RIGHT_HAND 74': 9,     # KEY_J
    'RIGHT_HAND 75': 9,     # KEY_K
    'RIGHT_HAND 76': 13,    # KEY_L
    'RIGHT_HAND 77': 10,    # KEY_M
    'RIGHT_HAND 78': 7,     # KEY_N
    'RIGHT_HAND 79': 12,    # KEY_O
    'RIGHT_HAND 80': 12,    # KEY_P
    'RIGHT_HAND 81': 17,    # KEY_Q
    'RIGHT_HAND 82': 2,     # KEY_R
    'RIGHT_HAND 83': 18,    # KEY_S
    'RIGHT_HAND 84': 5,     # KEY_T
    'RIGHT_HAND 85': 8,     # KEY_U
    'RIGHT_HAND 86': 4,     # KEY_V
    'RIGHT_HAND 87': 17,    # KEY_W
    'RIGHT_HAND 88': 19,    # KEY_X
    'RIGHT_HAND 89': 5,     # KEY_Y
    'RIGHT_HAND 90': 19,    # KEY_Z
    'RIGHT_HAND 262': 20,   # KEY_RIGHT
    'RIGHT_HAND 263': 20,   # KEY_LEFT
    'RIGHT_HAND 264': 21,   # KEY_DOWN
    'RIGHT_HAND 265': 21,   # KEY_UP
    'RIGHT_HAND 266': 22,   # KEY_PAGE_UP
    'RIGHT_HAND 267': 22,   # KEY_PAGE_DOWN
}


def keyboard(window, key, scancode, act, mods):
    global hand_in_use

    ######################################
    # kai = Keyboard-Actuation Interface #
    ######################################

    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        if hand_in_use == 'LEFT_HAND ':
            hand_in_use = 'RIGHT_HAND '
        else:
            hand_in_use = 'LEFT_HAND '

    # if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
    #     mj.mj_resetData(model, data)
    #     mj.mj_forward(model, data)

    if act == glfw.PRESS and key == glfw.KEY_ENTER:
        print(data.ctrl[:23])
        print(data.ctrl[23:46])
        print()

    if act == glfw.PRESS and key == glfw.KEY_LEFT_SHIFT:
        data.ctrl[:23] = gcsharpe

    if act == glfw.PRESS and key == glfw.KEY_RIGHT_SHIFT:
        data.ctrl[:23] = gaasharpd_init

    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        data.ctrl[kai[hand_in_use+str(key)]] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        data.ctrl[kai[hand_in_use+str(key)]] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_UP:
        data.ctrl[kai[hand_in_use+str(key)]] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_DOWN:
        data.ctrl[kai[hand_in_use+str(key)]] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_PAGE_UP:
        data.ctrl[kai[hand_in_use+str(key)]] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_PAGE_DOWN:
        data.ctrl[kai[hand_in_use+str(key)]] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_KP_0:
        data.ctrl[:23] = gcsharpe

    if act == glfw.PRESS and key == glfw.KEY_KP_1:
        data.ctrl[:23] = gcsharpe_g

    if act == glfw.PRESS and key == glfw.KEY_KP_2:
        data.ctrl[:23] = gcsharpe_c

    if act == glfw.PRESS and key == glfw.KEY_KP_3:
        data.ctrl[:23] = gcsharpe_e

    if act == glfw.PRESS and key == glfw.KEY_KP_DECIMAL:
        data.ctrl[:23] = gce

    if act == glfw.PRESS and key == glfw.KEY_KP_4:
        data.ctrl[:23] = gce_g

    if act == glfw.PRESS and key == glfw.KEY_KP_5:
        data.ctrl[:23] = gce_c

    if act == glfw.PRESS and key == glfw.KEY_KP_6:
        data.ctrl[:23] = gce_e

    if act == glfw.PRESS and key == glfw.KEY_KP_7:
        data.ctrl[:23] = gcdef_g3

    if act == glfw.PRESS and key == glfw.KEY_KP_8:
        data.ctrl[:23] = gcdef_c_sharp2

    if act == glfw.PRESS and key == glfw.KEY_KP_9:
        data.ctrl[:23] = gcdef_d_sharp1

    if act == glfw.PRESS and key == glfw.KEY_KP_DIVIDE:
        data.ctrl[:23] = gcdef_g4

    if act == glfw.PRESS and key == glfw.KEY_KP_MULTIPLY:
        data.ctrl[:23] = gcdef_c2

    if act == glfw.PRESS and key == glfw.KEY_KP_SUBTRACT:
        data.ctrl[:23] = gcdef_d_sharp2

    if hand_in_use == 'RIGHT_HAND ':
        if act == glfw.PRESS and key == glfw.KEY_1:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_2:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_3:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_4:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_5:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_6:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_7:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_8:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_9:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_0:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_Q:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_W:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_E:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_R:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_T:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_Y:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_U:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_I:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_O:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_P:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_A:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_S:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_D:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_F:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_G:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_H:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_J:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_K:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_L:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_Z:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_X:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_C:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_V:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_B:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_N:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_M:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_COMMA:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_PERIOD:
            data.ctrl[kai[hand_in_use+str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_SLASH:
            data.ctrl[kai[hand_in_use+str(key)]] -= 0.1
    elif hand_in_use == 'LEFT_HAND ':
        if act == glfw.PRESS and key == glfw.KEY_1:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_2:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_3:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_4:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_5:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_6:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_7:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_8:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_9:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_0:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_Q:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_W:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_E:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_R:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_T:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_Y:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_U:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_I:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_O:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_P:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_A:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_S:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_D:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_F:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_G:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_H:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_J:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_K:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_L:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_SEMICOLON:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_Z:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_X:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_C:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_V:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_B:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_N:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_M:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_COMMA:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1

        if act == glfw.PRESS and key == glfw.KEY_PERIOD:
            data.ctrl[kai[hand_in_use + str(key)]] += 0.1

        if act == glfw.PRESS and key == glfw.KEY_SLASH:
            data.ctrl[kai[hand_in_use + str(key)]] -= 0.1
    else:
        raise ValueError('Variable "hand_in_use" is neither "RIGHT_HAND " nor "LEFT_HAND "')

    data.ctrl[:46] = np.around(data.ctrl[:46], 3)


def mouse_button(window, button, act, mods):
    # update button state
    global button_left
    global button_middle
    global button_right

    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)

    # update mouse position
    glfw.get_cursor_pos(window)


def mouse_move(window, xpos, ypos):
    # compute mouse displacement, save
    global lastx
    global lasty
    global button_left
    global button_middle
    global button_right

    dx = xpos - lastx
    dy = ypos - lasty
    lastx = xpos
    lasty = ypos

    # no buttons down: nothing to do
    if (not button_left) and (not button_middle) and (not button_right):
        return

    # get current window size
    width, height = glfw.get_window_size(window)

    # get shift key state
    PRESS_LEFT_SHIFT = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
    PRESS_RIGHT_SHIFT = glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
    mod_shift = (PRESS_LEFT_SHIFT or PRESS_RIGHT_SHIFT)

    # determine action based on mouse button
    if button_right:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_MOVE_H
        else:
            action = mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        if mod_shift:
            action = mj.mjtMouse.mjMOUSE_ROTATE_H
        else:
            action = mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx / height, dy / height, scene, cam)


def scroll(window, xoffset, yoffset):
    action = mj.mjtMouse.mjMOUSE_ZOOM
    mj.mjv_moveCamera(model, action, 0.0, -0.05 * yoffset, scene, cam)


# get the full path
dirname = os.path.dirname(__file__)
abspath = os.path.join(dirname + "/" + xml_path)
xml_path = abspath

# MuJoCo data structures
model = mj.MjModel.from_xml_path(xml_path)  # MuJoCo model
data = mj.MjData(model)  # MuJoCo data
cam = mj.MjvCamera()  # Abstract camera
opt = mj.MjvOption()  # visualization options

# Init GLFW, create window, make OpenGL context current, request v-sync
glfw.init()
window = glfw.create_window(1200, 900, "Demo", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

# initialize visualization data structures
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# install GLFW mouse and keyboard callbacks
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)

cam.azimuth = -133.2000000000003
cam.elevation = -38.20000000000009
cam.distance = 1.428163427281391
cam.lookat = np.array([-0.3986208810237568, 0.1823730189635914, 1.1746658693793246])

# initialize the controller
init_controller(model, data)

# set the controller
mj.set_mjcb_control(controller)


async def side():
    pressed = [False] * 108
    pressed_threshold = 1
    counter1 = data.time
    counter2 = data.time
    # data.ctrl[left_hand] = a1_p
    while not glfw.window_should_close(window):
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:
            mj.mj_step(model, data)

        if data.time - counter1 > 0.05:
            counter1 = data.time

            if data.site_xpos[20][2] < pressed_threshold and not pressed[20]:
                pressed[20] = True
                sample = synth(frequencies_dict["A0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[21][2] < pressed_threshold and not pressed[21]:
                pressed[21] = True
                sample = synth(frequencies_dict["A#0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[22][2] < pressed_threshold and not pressed[22]:
                pressed[22] = True
                sample = synth(frequencies_dict["B0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[23][2] < pressed_threshold and not pressed[23]:
                pressed[23] = True
                sample = synth(frequencies_dict["C1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[24][2] < pressed_threshold and not pressed[24]:
                pressed[24] = True
                sample = synth(frequencies_dict["C#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[25][2] < pressed_threshold and not pressed[25]:
                pressed[25] = True
                sample = synth(frequencies_dict["D1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[26][2] < pressed_threshold and not pressed[26]:
                pressed[26] = True
                sample = synth(frequencies_dict["D#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[27][2] < pressed_threshold and not pressed[27]:
                pressed[27] = True
                sample = synth(frequencies_dict["E1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[28][2] < pressed_threshold and not pressed[28]:
                pressed[28] = True
                sample = synth(frequencies_dict["F1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[29][2] < pressed_threshold and not pressed[29]:
                pressed[29] = True
                sample = synth(frequencies_dict["F#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[30][2] < pressed_threshold and not pressed[30]:
                pressed[30] = True
                sample = synth(frequencies_dict["G1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[31][2] < pressed_threshold and not pressed[31]:
                pressed[31] = True
                sample = synth(frequencies_dict["G#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[32][2] < pressed_threshold and not pressed[32]:
                pressed[32] = True
                sample = synth(frequencies_dict["A1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[33][2] < pressed_threshold and not pressed[33]:
                pressed[33] = True
                sample = synth(frequencies_dict["A#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[34][2] < pressed_threshold and not pressed[34]:
                pressed[34] = True
                sample = synth(frequencies_dict["B1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[35][2] < pressed_threshold and not pressed[35]:
                pressed[35] = True
                sample = synth(frequencies_dict["C2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[36][2] < pressed_threshold and not pressed[36]:
                pressed[36] = True
                sample = synth(frequencies_dict["C#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[37][2] < pressed_threshold and not pressed[37]:
                pressed[37] = True
                sample = synth(frequencies_dict["D2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[38][2] < pressed_threshold and not pressed[38]:
                pressed[38] = True
                sample = synth(frequencies_dict["D#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[39][2] < pressed_threshold and not pressed[39]:
                pressed[39] = True
                sample = synth(frequencies_dict["E2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[40][2] < pressed_threshold and not pressed[40]:
                pressed[40] = True
                sample = synth(frequencies_dict["F2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[41][2] < pressed_threshold and not pressed[41]:
                pressed[41] = True
                sample = synth(frequencies_dict["F#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[42][2] < pressed_threshold and not pressed[42]:
                pressed[42] = True
                sample = synth(frequencies_dict["G2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[43][2] < pressed_threshold and not pressed[43]:
                pressed[43] = True
                sample = synth(frequencies_dict["G#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[44][2] < pressed_threshold and not pressed[44]:
                pressed[44] = True
                sample = synth(frequencies_dict["A2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[45][2] < pressed_threshold and not pressed[45]:
                pressed[45] = True
                sample = synth(frequencies_dict["A#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[46][2] < pressed_threshold and not pressed[46]:
                pressed[46] = True
                sample = synth(frequencies_dict["B2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[47][2] < pressed_threshold and not pressed[47]:
                pressed[47] = True
                sample = synth(frequencies_dict["C3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[48][2] < pressed_threshold and not pressed[48]:
                pressed[48] = True
                sample = synth(frequencies_dict["C#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[49][2] < pressed_threshold and not pressed[49]:
                pressed[49] = True
                sample = synth(frequencies_dict["D3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[50][2] < pressed_threshold and not pressed[50]:
                pressed[50] = True
                sample = synth(frequencies_dict["D#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[51][2] < pressed_threshold and not pressed[51]:
                pressed[51] = True
                sample = synth(frequencies_dict["E3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[52][2] < pressed_threshold and not pressed[52]:
                pressed[52] = True
                sample = synth(frequencies_dict["F3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[53][2] < pressed_threshold and not pressed[53]:
                pressed[53] = True
                sample = synth(frequencies_dict["F#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[54][2] < pressed_threshold and not pressed[54]:
                pressed[54] = True
                sample = synth(frequencies_dict["G3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[55][2] < pressed_threshold and not pressed[55]:
                pressed[55] = True
                sample = synth(frequencies_dict["G#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[56][2] < pressed_threshold and not pressed[56]:
                pressed[56] = True
                sample = synth(frequencies_dict["A3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[57][2] < pressed_threshold and not pressed[57]:
                pressed[57] = True
                sample = synth(frequencies_dict["A#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[58][2] < pressed_threshold and not pressed[58]:
                pressed[58] = True
                sample = synth(frequencies_dict["B3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[59][2] < pressed_threshold and not pressed[59]:
                pressed[59] = True
                sample = synth(frequencies_dict["C4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[60][2] < pressed_threshold and not pressed[60]:
                pressed[60] = True
                sample = synth(frequencies_dict["C#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[61][2] < pressed_threshold and not pressed[61]:
                pressed[61] = True
                sample = synth(frequencies_dict["D4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[62][2] < pressed_threshold and not pressed[62]:
                pressed[62] = True
                sample = synth(frequencies_dict["D#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[63][2] < pressed_threshold and not pressed[63]:
                pressed[63] = True
                sample = synth(frequencies_dict["E4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[64][2] < pressed_threshold and not pressed[64]:
                pressed[64] = True
                sample = synth(frequencies_dict["F4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[65][2] < pressed_threshold and not pressed[65]:
                pressed[65] = True
                sample = synth(frequencies_dict["F#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[66][2] < pressed_threshold and not pressed[66]:
                pressed[66] = True
                sample = synth(frequencies_dict["G4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[67][2] < pressed_threshold and not pressed[67]:
                pressed[67] = True
                sample = synth(frequencies_dict["G#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[68][2] < pressed_threshold and not pressed[68]:
                pressed[68] = True
                sample = synth(frequencies_dict["A4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[69][2] < pressed_threshold and not pressed[69]:
                pressed[69] = True
                sample = synth(frequencies_dict["A#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[70][2] < pressed_threshold and not pressed[70]:
                pressed[70] = True
                sample = synth(frequencies_dict["B4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[71][2] < pressed_threshold and not pressed[71]:
                pressed[71] = True
                sample = synth(frequencies_dict["C5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[72][2] < pressed_threshold and not pressed[72]:
                pressed[72] = True
                sample = synth(frequencies_dict["C#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[73][2] < pressed_threshold and not pressed[73]:
                pressed[73] = True
                sample = synth(frequencies_dict["D5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[74][2] < pressed_threshold and not pressed[74]:
                pressed[74] = True
                sample = synth(frequencies_dict["D#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[75][2] < pressed_threshold and not pressed[75]:
                pressed[75] = True
                sample = synth(frequencies_dict["E5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[76][2] < pressed_threshold and not pressed[76]:
                pressed[76] = True
                sample = synth(frequencies_dict["F5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[77][2] < pressed_threshold and not pressed[77]:
                pressed[77] = True
                sample = synth(frequencies_dict["F#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[78][2] < pressed_threshold and not pressed[78]:
                pressed[78] = True
                sample = synth(frequencies_dict["G5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[79][2] < pressed_threshold and not pressed[79]:
                pressed[79] = True
                sample = synth(frequencies_dict["G#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[80][2] < pressed_threshold and not pressed[80]:
                pressed[80] = True
                sample = synth(frequencies_dict["A5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[81][2] < pressed_threshold and not pressed[81]:
                pressed[81] = True
                sample = synth(frequencies_dict["A#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[82][2] < pressed_threshold and not pressed[82]:
                pressed[82] = True
                sample = synth(frequencies_dict["B5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[83][2] < pressed_threshold and not pressed[83]:
                pressed[83] = True
                sample = synth(frequencies_dict["C6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[84][2] < pressed_threshold and not pressed[84]:
                pressed[84] = True
                sample = synth(frequencies_dict["C#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[85][2] < pressed_threshold and not pressed[85]:
                pressed[85] = True
                sample = synth(frequencies_dict["D6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[86][2] < pressed_threshold and not pressed[86]:
                pressed[86] = True
                sample = synth(frequencies_dict["D#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[87][2] < pressed_threshold and not pressed[87]:
                pressed[87] = True
                sample = synth(frequencies_dict["E6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[88][2] < pressed_threshold and not pressed[88]:
                pressed[88] = True
                sample = synth(frequencies_dict["F6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[89][2] < pressed_threshold and not pressed[89]:
                pressed[89] = True
                sample = synth(frequencies_dict["F#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[90][2] < pressed_threshold and not pressed[90]:
                pressed[90] = True
                sample = synth(frequencies_dict["G6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[91][2] < pressed_threshold and not pressed[91]:
                pressed[91] = True
                sample = synth(frequencies_dict["G#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[92][2] < pressed_threshold and not pressed[92]:
                pressed[92] = True
                sample = synth(frequencies_dict["A6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[93][2] < pressed_threshold and not pressed[93]:
                pressed[93] = True
                sample = synth(frequencies_dict["A#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[94][2] < pressed_threshold and not pressed[94]:
                pressed[94] = True
                sample = synth(frequencies_dict["B6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[95][2] < pressed_threshold and not pressed[95]:
                pressed[95] = True
                sample = synth(frequencies_dict["C7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[96][2] < pressed_threshold and not pressed[96]:
                pressed[96] = True
                sample = synth(frequencies_dict["C#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[97][2] < pressed_threshold and not pressed[97]:
                pressed[97] = True
                sample = synth(frequencies_dict["D7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[98][2] < pressed_threshold and not pressed[98]:
                pressed[98] = True
                sample = synth(frequencies_dict["D#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[99][2] < pressed_threshold and not pressed[99]:
                pressed[99] = True
                sample = synth(frequencies_dict["E7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[100][2] < pressed_threshold and not pressed[100]:
                pressed[100] = True
                sample = synth(frequencies_dict["F7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[101][2] < pressed_threshold and not pressed[101]:
                pressed[101] = True
                sample = synth(frequencies_dict["F#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[102][2] < pressed_threshold and not pressed[102]:
                pressed[102] = True
                sample = synth(frequencies_dict["G7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[103][2] < pressed_threshold and not pressed[103]:
                pressed[103] = True
                sample = synth(frequencies_dict["G#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[104][2] < pressed_threshold and not pressed[104]:
                pressed[104] = True
                sample = synth(frequencies_dict["A7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[105][2] < pressed_threshold and not pressed[105]:
                pressed[105] = True
                sample = synth(frequencies_dict["A#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[106][2] < pressed_threshold and not pressed[106]:
                pressed[106] = True
                sample = synth(frequencies_dict["B7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[107][2] < pressed_threshold and not pressed[107]:
                pressed[107] = True
                sample = synth(frequencies_dict["C8"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

        if data.time - counter2 > 0.05:
            counter2 = data.time

            if data.site_xpos[20][2] > pressed_threshold:
                pressed[20] = False
            if data.site_xpos[21][2] > pressed_threshold:
                pressed[21] = False
            if data.site_xpos[22][2] > pressed_threshold:
                pressed[22] = False

            if data.site_xpos[23][2] > pressed_threshold:
                pressed[23] = False
            if data.site_xpos[24][2] > pressed_threshold:
                pressed[24] = False
            if data.site_xpos[25][2] > pressed_threshold:
                pressed[25] = False
            if data.site_xpos[26][2] > pressed_threshold:
                pressed[26] = False
            if data.site_xpos[27][2] > pressed_threshold:
                pressed[27] = False
            if data.site_xpos[28][2] > pressed_threshold:
                pressed[28] = False
            if data.site_xpos[29][2] > pressed_threshold:
                pressed[29] = False
            if data.site_xpos[30][2] > pressed_threshold:
                pressed[30] = False
            if data.site_xpos[31][2] > pressed_threshold:
                pressed[31] = False
            if data.site_xpos[32][2] > pressed_threshold:
                pressed[32] = False
            if data.site_xpos[33][2] > pressed_threshold:
                pressed[33] = False
            if data.site_xpos[34][2] > pressed_threshold:
                pressed[34] = False

            if data.site_xpos[35][2] > pressed_threshold:
                pressed[35] = False
            if data.site_xpos[36][2] > pressed_threshold:
                pressed[36] = False
            if data.site_xpos[37][2] > pressed_threshold:
                pressed[37] = False
            if data.site_xpos[38][2] > pressed_threshold:
                pressed[38] = False
            if data.site_xpos[39][2] > pressed_threshold:
                pressed[39] = False
            if data.site_xpos[40][2] > pressed_threshold:
                pressed[40] = False
            if data.site_xpos[41][2] > pressed_threshold:
                pressed[41] = False
            if data.site_xpos[42][2] > pressed_threshold:
                pressed[42] = False
            if data.site_xpos[43][2] > pressed_threshold:
                pressed[43] = False
            if data.site_xpos[44][2] > pressed_threshold:
                pressed[44] = False
            if data.site_xpos[45][2] > pressed_threshold:
                pressed[45] = False
            if data.site_xpos[46][2] > pressed_threshold:
                pressed[46] = False

            if data.site_xpos[47][2] > pressed_threshold:
                pressed[47] = False
            if data.site_xpos[48][2] > pressed_threshold:
                pressed[48] = False
            if data.site_xpos[49][2] > pressed_threshold:
                pressed[49] = False
            if data.site_xpos[50][2] > pressed_threshold:
                pressed[50] = False
            if data.site_xpos[51][2] > pressed_threshold:
                pressed[51] = False
            if data.site_xpos[52][2] > pressed_threshold:
                pressed[52] = False
            if data.site_xpos[53][2] > pressed_threshold:
                pressed[53] = False
            if data.site_xpos[54][2] > pressed_threshold:
                pressed[54] = False
            if data.site_xpos[55][2] > pressed_threshold:
                pressed[55] = False
            if data.site_xpos[56][2] > pressed_threshold:
                pressed[56] = False
            if data.site_xpos[57][2] > pressed_threshold:
                pressed[57] = False
            if data.site_xpos[58][2] > pressed_threshold:
                pressed[58] = False

            if data.site_xpos[59][2] > pressed_threshold:
                pressed[59] = False
            if data.site_xpos[60][2] > pressed_threshold:
                pressed[60] = False
            if data.site_xpos[61][2] > pressed_threshold:
                pressed[61] = False
            if data.site_xpos[62][2] > pressed_threshold:
                pressed[62] = False
            if data.site_xpos[63][2] > pressed_threshold:
                pressed[63] = False
            if data.site_xpos[64][2] > pressed_threshold:
                pressed[64] = False
            if data.site_xpos[65][2] > pressed_threshold:
                pressed[65] = False
            if data.site_xpos[66][2] > pressed_threshold:
                pressed[66] = False
            if data.site_xpos[67][2] > pressed_threshold:
                pressed[67] = False
            if data.site_xpos[68][2] > pressed_threshold:
                pressed[68] = False
            if data.site_xpos[69][2] > pressed_threshold:
                pressed[69] = False
            if data.site_xpos[70][2] > pressed_threshold:
                pressed[70] = False

            if data.site_xpos[71][2] > pressed_threshold:
                pressed[71] = False
            if data.site_xpos[72][2] > pressed_threshold:
                pressed[72] = False
            if data.site_xpos[73][2] > pressed_threshold:
                pressed[73] = False
            if data.site_xpos[74][2] > pressed_threshold:
                pressed[74] = False
            if data.site_xpos[75][2] > pressed_threshold:
                pressed[75] = False
            if data.site_xpos[76][2] > pressed_threshold:
                pressed[76] = False
            if data.site_xpos[77][2] > pressed_threshold:
                pressed[77] = False
            if data.site_xpos[78][2] > pressed_threshold:
                pressed[78] = False
            if data.site_xpos[79][2] > pressed_threshold:
                pressed[79] = False
            if data.site_xpos[80][2] > pressed_threshold:
                pressed[80] = False
            if data.site_xpos[81][2] > pressed_threshold:
                pressed[81] = False
            if data.site_xpos[82][2] > pressed_threshold:
                pressed[82] = False

            if data.site_xpos[83][2] > pressed_threshold:
                pressed[83] = False
            if data.site_xpos[84][2] > pressed_threshold:
                pressed[84] = False
            if data.site_xpos[85][2] > pressed_threshold:
                pressed[85] = False
            if data.site_xpos[86][2] > pressed_threshold:
                pressed[86] = False
            if data.site_xpos[87][2] > pressed_threshold:
                pressed[87] = False
            if data.site_xpos[88][2] > pressed_threshold:
                pressed[88] = False
            if data.site_xpos[89][2] > pressed_threshold:
                pressed[89] = False
            if data.site_xpos[90][2] > pressed_threshold:
                pressed[90] = False
            if data.site_xpos[91][2] > pressed_threshold:
                pressed[91] = False
            if data.site_xpos[92][2] > pressed_threshold:
                pressed[92] = False
            if data.site_xpos[93][2] > pressed_threshold:
                pressed[93] = False
            if data.site_xpos[94][2] > pressed_threshold:
                pressed[94] = False

            if data.site_xpos[95][2] > pressed_threshold:
                pressed[95] = False
            if data.site_xpos[96][2] > pressed_threshold:
                pressed[96] = False
            if data.site_xpos[97][2] > pressed_threshold:
                pressed[97] = False
            if data.site_xpos[98][2] > pressed_threshold:
                pressed[98] = False
            if data.site_xpos[99][2] > pressed_threshold:
                pressed[99] = False
            if data.site_xpos[100][2] > pressed_threshold:
                pressed[100] = False
            if data.site_xpos[101][2] > pressed_threshold:
                pressed[101] = False
            if data.site_xpos[102][2] > pressed_threshold:
                pressed[102] = False
            if data.site_xpos[103][2] > pressed_threshold:
                pressed[103] = False
            if data.site_xpos[104][2] > pressed_threshold:
                pressed[104] = False
            if data.site_xpos[105][2] > pressed_threshold:
                pressed[105] = False
            if data.site_xpos[106][2] > pressed_threshold:
                pressed[106] = False

            if data.site_xpos[107][2] > pressed_threshold:
                pressed[107] = False

        viewport_width, viewport_height = glfw.get_framebuffer_size(window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # print camera configuration (help to initialize the view)
        if print_camera_config == 1:
            print('cam.azimuth =', cam.azimuth, ';', 'cam.elevation =', cam.elevation, ';', 'cam.distance = ',
                  cam.distance)
            print('cam.lookat =np.array([', cam.lookat[0], ',', cam.lookat[1], ',', cam.lookat[2], '])')

        # Update scene and render
        mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
        mj.mjr_render(viewport, scene, context)

        # swap OpenGL buffers (blocking call due to v-sync)
        glfw.swap_buffers(window)

        # process pending GUI events, call GLFW callbacks
        glfw.poll_events()

        await asyncio.sleep(0.0001)

# Left Hand
csharp2_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,    0. ,   0. ,   0. ,  0.  ,  0.  , -0.1,  0.1,
                        -0.3,   0.4,  0.3,   0.,    0.3,   -0.1,  -0.4,   0.7, -0.24,  0.17,  0. ,])
csharp2_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,    0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                        -0.3,   0.4,  0.3,   0.,    0.3,   -0.1,  -0.4,   0.7, -0.24,  0.17, -0.07,])
b1_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.   ,  0.  , -0.1 ,  0.1,
                   -0.3,   0.4,  0.3,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.175,  0.12, -0.02,])
b1_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.   ,  0.  , -0.1 ,  0.1,
                   -0.3,   0.4,  0.3,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.175,  0.12, -0.08,])
a1_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                   -0.3,   0.4,  0.3,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.15,  0.12, -0.02,])
a1_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                   -0.3,   0.4,  0.3,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.15,  0.12, -0.08,])
fsharp1_trans = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1, 0.1,
                            -0.3,   0.4,  0.3,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.11,  0.05,  0.1,])
fsharp1_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.11,  0.17,  0. ,])
fsharp1_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.11,  0.17, -0.07,])
gsharp1_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.14,  0.17,  0. ,])
gsharp1_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.14,  0.17, -0.07,])
csharp1_f = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.04,  0.17,  0. ,])
csharp1_p = np.asarray([-0.3,  -0.3,  0. ,   0.,    0. ,   0. ,   0. ,   0. ,  0.  ,  0.  , -0.1 ,  0.1,
                        -0.3,   0.4,  0.5,   0.,    0.3,  -0.1,  -0.4,   0.7, -0.04,  0.17, -0.07,])

# Right Hand
gcsharpe_init = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.15, 0.5, 0.4, 0.2, 0.65, 0.3, 0.,
                            0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.02, ])
gcsharpe = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.15, 0.5, 0.4, 0.2, 0.65, 0.3, 0.,
                       0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
gcsharpe_g = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.1, 0.5, 0.4, 0.2, 0.7, 0.3, 0.,
                         0.1, 0.4, 0.4, 0., 0.6, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
gcsharpe_c = np.asarray([-0.3, -0.3, 0.25, 0.7, 0.5, 0.1, 0.5, 0.4, 0.2, 0.7, 0.3, 0.,
                         0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])
gcsharpe_e = np.asarray([-0.3, -0.3, 0.25, 0.4, 0.5, 0.1, 0.5, 0.4, 0.2, 0.9, 0.3, 0.,
                         0.1, 0.4, 0.4, 0., 0.3, 0., 0.1, 0., 0.08, 0.16, -0.04, ])

gce_trans = np.asarray([-0.3, -0.3, 0.1, 0.8, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                        -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gce = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                  -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gce_g = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                    -0.05, 0.5, 0.2, -0.2, 0.6, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gce_c = np.asarray([-0.3, -0.3, 0.1, 0.6, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                    -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gce_e = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.7, 0.7, 0.,
                    -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gdf_d = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.8, 0.4, -0.05, 0.4, 0.7, 0.,
                    -0.05, 0.5, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gdf_d_trans = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.8, 0.4, -0.05, 0.4, 0.7, 0.,
                          -0.05, 0.5, 0.2, -0.2, 0.3, 0., -0.5, 0., 0.09, 0.13, -0.04, ])
gdf_f = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, 0.15, 0.6, 0.4, -0.05, 0.4, 0.7, 0.,
                    -0.05, 0.8, 0.2, -0.2, 0.4, 0., -0.5, 0., 0.09, 0.13, -0.04, ])

gcdef_trans = np.asarray([-0.3, -0.3, 0.3, 0.5, 0.5, 0.15, 0.75, 0.5, 0., 0.5, 0.5, 0.,
                          -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_g1 = np.asarray([-0.3, -0.3, 0.3, 0.5, 0.5, 0.15, 0.25, 0.5, 0., 0.5, 0.5, 0.,
                       -0.1, 0.2, 0.6, -0.1, 0.55, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_c1 = np.asarray([-0.3, -0.3, 0.3, 0.7, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                       -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_f = np.asarray([-0.3, -0.3, 0.3, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                      -0.1, 0.5, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_g2 = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                       -0.1, 0.2, 0.6, -0.1, 0.5, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_c_sharp1 = np.asarray([-0.3, -0.3, 0.1, 0.6, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                             -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_e = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.7, 0.5, 0.,
                      -0.1, 0.2, 0.6, -0.1, 0.3, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
gcdef_g3 = np.asarray([-0.3, -0.3, 0.1, 0.3, 0.5, -0.05, 0.35, 0.5, 0., 0.5, 0.5, 0.,
                       -0.1, 0.2, 0.6, -0.1, 0.5, 0., 0., -0.5, 0.09, 0.15, -0.04, ])
# gcdef_c_sharp2 = np.asarray([-0.3,  -0.3,   0.1,   0.6,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
# gcdef_d_sharp1 = np.asarray([-0.3,  -0.3,   0.3,   0.3,   0.5,   -0.05,  0.55,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
# gcdef_g4 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                        -0.1,   0.2,   0.6,  -0.1,   0.5,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
# gcdef_c2 = np.asarray([-0.3,  -0.3,   0.3,   0.7,   0.5,   -0.05,  0.35,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                        -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])
# gcdef_d_sharp2 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,   -0.05,  0.55,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                              -0.1,   0.2,   0.6,  -0.1,   0.3,    0.  ,  0.  ,  -0.5,   0.09,  0.15, -0.04,])

#
# gcdef = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,  -0.05,  0.4,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                     -0.1,   0.2,   0.6,  -0.1,   0.3,   0.  ,  0. ,  -0.4,   0.09,  0.15, -0.04,])
# gcdef_g2 = np.asarray([-0.3,  -0.3,   0.3,   0.5,   0.5,  -0.05,  0.4,   0.5,   0.  ,  0.5 ,  0.5 ,  0.,
#                     -0.1,   0.2,   0.6,  -0.1,   0.3,   0.  ,  0. ,  -0.4,   0.09,  0.15, -0.04,])
gcdef_c_sharp2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             2.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                             9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
gcdef_d_sharp1 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 3.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 5.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                             9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
gcdef_g4 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                       5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                       0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                       -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                       6.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                       9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
gcdef_c2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 7.00000000e-01,
                       5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                       0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                       -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                       3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                       9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])
gcdef_d_sharp2 = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                             5.00000000e-01, -5.00000000e-02, 5.50000000e-01, 5.00000000e-01,
                             0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                             -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                             3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                             9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])

dfsharpc_init = np.asarray([-0.2 , -0.3,   0.3,   0.2,   0.5,  -0.05,  0.15,  0.5,   0.  ,  0.1 ,  0.5 ,  0.,
                             0.05,  0.1,   0.1,  -0.6,   0.1,   0.  , -0.5 , -0.2,   0.15,  0.16, -0.02,])
dfsharpc = np.asarray([-0.2, -0.3,   0.3,   0.2,   0.5,  -0.05,  0.15,  0.5,   0.  ,  0.1 ,  0.5 ,  0.,
                        0.1,  0.3,   0.6,  -0.6,   0.1,   0.  , -0.5 , -0.2,   0.15,  0.16, -0.04,])
dfsharpc_d = np.asarray([-2.00000000e-01, -3.00000000e-01,  3.00000000e-01,  2.00000000e-01,
                          5.00000000e-01, -5.00000000e-02,  1.50000000e-01,  5.00000000e-01,
                          0.00000000e+00,  1.00000000e-01,  5.00000000e-01,  0.00000000e+00,
                         -5.00000000e-02,  3.00000000e-01,  6.00000000e-01, -6.00000000e-01,
                          4.00000000e-01,  0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                          1.50000000e-01,  1.60000000e-01, -4.00000000e-02,])
dfsharpc_f = np.asarray([-2.00000000e-01, -3.00000000e-01,  3.00000000e-01,  6.00000000e-01,
                          5.00000000e-01, -5.00000000e-02,  1.50000000e-01,  5.00000000e-01,
                          0.00000000e+00,  1.00000000e-01,  5.00000000e-01,  0.00000000e+00,
                         -5.00000000e-02,  3.00000000e-01,  6.00000000e-01, -6.00000000e-01,
                          1.00000000e-01,  0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                          1.50000000e-01,  1.60000000e-01, -4.00000000e-02,])
dfsharpc_c = np.asarray([-2.00000000e-01, -3.00000000e-01,  3.00000000e-01,  3.00000000e-01,
                          5.00000000e-01, -5.00000000e-02,  1.50000000e-01,  5.00000000e-01,
                          0.00000000e+00,  1.00000000e-01,  5.00000000e-01,  0.00000000e+00,
                         -5.00000000e-02,  6.00000000e-01,  6.00000000e-01, -6.00000000e-01,
                          1.00000000e-01,  0.00000000e+00, -5.00000000e-01, -2.00000000e-01,
                          1.50000000e-01,  1.60000000e-01, -4.00000000e-02,])

end = np.asarray([-3.00000000e-01, -3.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                  5.00000000e-01, -5.00000000e-02, 3.50000000e-01, 5.00000000e-01,
                  0.00000000e+00, 5.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                  -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, -6.00000000e-01,
                  3.00000000e-01, 0.00000000e+00, 2.77555756e-17, -2.77555756e-17,
                  9.00000000e-02, 1.60000000e-01, -4.00000000e-02, ])

beat = 0.4  # 150 BPM / 2.5 BPS / 0.4 SPB (B=Beats, P=Per, M=Minutes, S=Seconds)

whole = 4 * beat
half = 2 * beat
quarter = beat
eighth = beat / 2
sixteenth = beat / 4


left_hand = slice(23, 46)
right_hand = slice(0, 23)


async def sequence():
    await asyncio.sleep(1)
    data.ctrl[right_hand] = gcsharpe_init
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(1)
    data.ctrl[right_hand] = gcsharpe
    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(sixteenth)

    # First Position (First Part)
    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcsharpe_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcsharpe_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_trans
    data.ctrl[left_hand] = a1_f
    await asyncio.sleep(quarter)

    # Second Position (First Part)
    data.ctrl[right_hand] = gce_g
    data.ctrl[left_hand] = a1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gce_c
    data.ctrl[left_hand] = fsharp1_f
    await asyncio.sleep(quarter)

    # Third-Second Position (First Part)
    data.ctrl[right_hand] = gce_g
    data.ctrl[left_hand] = fsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gce_g
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gdf_d_trans
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    # Last Position (First Part)
    data.ctrl[right_hand] = gcdef_trans
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = gcdef_g1
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c1
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_f
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g2
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c_sharp1
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_e
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g3
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c_sharp2
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_d_sharp1
    await asyncio.sleep(quarter)

    data.ctrl[right_hand] = gcdef_g4
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_c2
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = gcdef_d_sharp2
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    data.ctrl[right_hand] = dfsharpc_init
    await asyncio.sleep(eighth)
    data.ctrl[right_hand] = dfsharpc
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_d
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_f
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_c
    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[right_hand] = dfsharpc_d
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(eighth)

    await asyncio.sleep(1)
    data.ctrl[right_hand] = end


async def sequence_():
    await asyncio.sleep(1)
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(2)

    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = csharp2_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = csharp2_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = b1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = b1_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = a1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = a1_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = fsharp1_trans
    await asyncio.sleep(eighth)
    data.ctrl[left_hand] = fsharp1_f
    await asyncio.sleep(eighth)

    data.ctrl[left_hand] = fsharp1_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(7 * quarter)
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(half)
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(half)
    data.ctrl[left_hand] = gsharp1_f
    await asyncio.sleep(quarter)

    data.ctrl[left_hand] = gsharp1_p
    await asyncio.sleep(quarter)
    data.ctrl[left_hand] = csharp1_f
    await asyncio.sleep(quarter)


async def main():
    task1 = asyncio.create_task(side())
    task2 = asyncio.create_task(sequence())

    await task2
    await task1


asyncio.run(main())

glfw.terminate()
