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
    # arr = np.clip(arr*10, -1, 1) # squarish waves
    # arr = np.cumsum(np.clip(arr*10, -1, 1)) # triangularish waves pt1
    # arr = arr+np.sin(2*np.pi*frequency*np.linspace(0,duration, frames)) # triangularish waves pt1
    arr = arr / max(np.abs(arr))  # triangularish waves pt1
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


main_init = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
                        4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                        4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
                        0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
                        2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
                        -1.00000000e-01, 1.60000000e-01, -5.00000000e-02])
main_a_sharp = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
                           4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                           4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
                           0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
                           4.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
                           -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
main_d = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 6.00000000e-01,
                     4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                     4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
                     0.00000000e+00, 2.77555756e-17, 5.00000000e-01, 0.00000000e+00,
                     2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
                     -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
main_g = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
                     4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                     4.00000000e-01, 3.00000000e-01, 6.00000000e-01, 3.00000000e-01,
                     0.00000000e+00, 2.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                     2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
                     -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])
main_f = np.asarray([-4.00000000e-01, -3.00000000e-01, 4.00000000e-01, 4.00000000e-01,
                     4.00000000e-01, 4.00000000e-01, 1.00000000e-01, 6.00000000e-01,
                     4.00000000e-01, 5.00000000e-01, 6.00000000e-01, 3.00000000e-01,
                     0.00000000e+00, 1.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                     2.00000000e-01, 0.00000000e+00, -2.77555756e-17, -3.00000000e-01,
                     -1.00000000e-01, 1.60000000e-01, -5.00000000e-02, ])

gaasharpd_pre_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                                 6.00000000e-01, 4.00000000e-01, 5.00000000e-01, 2.00000000e-01,
                                 2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                                 -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                                 4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                                 -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
gaasharpd_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                             6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
                             2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                             -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                             4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                             -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
gaasharpd_g = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                          6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
                          2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          7.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                          -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
gaasharpd_a = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 7.00000000e-01,
                          6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
                          2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                          -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
gaasharpd_a_sharp = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                                6.00000000e-01, 4.00000000e-01, 9.00000000e-01, 2.00000000e-01,
                                2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                                -1.00000000e-01, 4.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                                4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                                -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
gaasharpd_d = np.asarray([-2.00000000e-01, -5.00000000e-01, 3.00000000e-01, 5.00000000e-01,
                          6.00000000e-01, 4.00000000e-01, 7.00000000e-01, 2.00000000e-01,
                          2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          -1.00000000e-01, 6.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                          4.00000000e-01, 1.00000000e-01, -6.00000000e-01, 2.77555756e-17,
                          -5.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])

acdef_init = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
                         6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
                         2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                         -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                         3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                         -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
acdef_a = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
                      6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
                      2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                      -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                      5.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                      -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
acdef_c = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 7.00000000e-01,
                      6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
                      2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                      -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                      3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                      -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
acdef_d = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
                      6.00000000e-01, 2.77555756e-17, 7.00000000e-01, 6.00000000e-01,
                      2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                      -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                      3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                      -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
acdef_e = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
                      6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
                      2.77555756e-17, 7.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                      -1.00000000e-01, 3.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                      3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                      -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])
acdef_f = np.asarray([-2.00000000e-01, -5.00000000e-01, 2.77555756e-17, 5.00000000e-01,
                      6.00000000e-01, 2.77555756e-17, 5.00000000e-01, 6.00000000e-01,
                      2.77555756e-17, 5.00000000e-01, 6.00000000e-01, 1.00000000e-01,
                      -1.00000000e-01, 6.00000000e-01, 6.00000000e-01, -2.00000000e-01,
                      3.00000000e-01, 1.00000000e-01, -4.00000000e-01, -3.00000000e-01,
                      -9.00000000e-02, 1.00000000e-01, -7.00000000e-02, ])


def quat2euler(quat_mujoco):
    # mujocoy quat is constant,x,y,z,
    # scipy quaut is x,y,z,constant
    quat_scipy = np.array([quat_mujoco[3], quat_mujoco[0], quat_mujoco[1], quat_mujoco[2]])

    r = R.from_quat(quat_scipy)
    euler = r.as_euler('xyz', degrees=True)

    return euler


def init_controller(model, data):
    # initialize the controller here. This function is called once, in the beginning
    data.ctrl[23:] = -0.5


# def repetitive_notes(repetitions, duration):
#     global time_list
#     for _ in range(repetitions):
#         time_list.append(duration)
#
# time_list = list()
# time_list.append(0.5)
# repetitive_notes(2, 0.25)
# time_list.append(0.5)
# time_list.append(0.1)
# repetitive_notes(6, 0.25)
# time_list.append(0.5)
# time_list.append(0.1)
# repetitive_notes(6, 0.25)
# time_list.append(0.5)
# time_list.append(0.1)
# repetitive_notes(6, 0.25)
# time_list.append(2.0)
# time_list.append(1.5)
# time_list.append(2.0)
# repetitive_notes(3, 0.25)
# time_list.append(0.5)
# time_list.append(2.0)
# repetitive_notes(6, 0.5)
# repetitive_notes(2, 5.0)

# new_list = list()
# # new_list.append(time_list[0])
#
# for index in range(1, len(time_list) + 1):
#     new_list.append(sum(time_list[:index]))
#
# time_list = np.asarray(new_list)

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


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

    if act == glfw.PRESS and key == glfw.KEY_ENTER:
        print(data.ctrl[:23])
        print()

    if act == glfw.PRESS and key == glfw.KEY_LEFT_SHIFT:
        data.ctrl[:23] = gcsharpe

    if act == glfw.PRESS and key == glfw.KEY_RIGHT_SHIFT:
        data.ctrl[:23] = gaasharpd_init

    if act == glfw.PRESS and key == glfw.KEY_LEFT:
        data.ctrl[20] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_RIGHT:
        data.ctrl[20] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_UP:
        data.ctrl[21] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_DOWN:
        data.ctrl[21] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_PAGE_UP:
        data.ctrl[22] += 0.01

    if act == glfw.PRESS and key == glfw.KEY_PAGE_DOWN:
        data.ctrl[22] -= 0.01

    if act == glfw.PRESS and key == glfw.KEY_KP_0:
        data.ctrl[:23] = gcdef_trans

    if act == glfw.PRESS and key == glfw.KEY_KP_1:
        data.ctrl[:23] = gcdef_g1

    if act == glfw.PRESS and key == glfw.KEY_KP_2:
        data.ctrl[:23] = gcdef_c1

    if act == glfw.PRESS and key == glfw.KEY_KP_3:
        data.ctrl[:23] = gcdef_f

    if act == glfw.PRESS and key == glfw.KEY_KP_4:
        data.ctrl[:23] = gcdef_g2

    if act == glfw.PRESS and key == glfw.KEY_KP_5:
        data.ctrl[:23] = gcdef_c_sharp1

    if act == glfw.PRESS and key == glfw.KEY_KP_6:
        data.ctrl[:23] = gcdef_e

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

    if act == glfw.PRESS and key == glfw.KEY_1:
        data.ctrl[15] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_2:
        data.ctrl[15] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_3:
        data.ctrl[16] += 0.1

    # if act == glfw.PRESS and key == glfw.KEY_0:
    #     data.ctrl[:20] = initial_pos
    #
    # if act == glfw.PRESS and key == glfw.KEY_1:
    #     data.ctrl[:20] = g_sharp_pos
    #
    # if act == glfw.PRESS and key == glfw.KEY_2:
    #     data.ctrl[:20] = c_sharp_pos
    #
    # if act == glfw.PRESS and key == glfw.KEY_3:
    #     data.ctrl[:20] = e_pos

    if act == glfw.PRESS and key == glfw.KEY_4:
        data.ctrl[16] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_5:
        data.ctrl[0] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_6:
        data.ctrl[0] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_7:
        data.ctrl[1] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_8:
        data.ctrl[1] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_9:
        data.ctrl[11] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_0:
        data.ctrl[11] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_Q:
        data.ctrl[17] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_W:
        data.ctrl[17] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_E:
        data.ctrl[2] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_R:
        data.ctrl[2] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_T:
        data.ctrl[5] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_Y:
        data.ctrl[5] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_U:
        data.ctrl[8] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_I:
        data.ctrl[8] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_O:
        data.ctrl[12] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_P:
        data.ctrl[12] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_LEFT_BRACKET:
        data.ctrl[10] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_RIGHT_BRACKET:
        data.ctrl[10] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_A:
        data.ctrl[18] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_S:
        data.ctrl[18] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_D:
        data.ctrl[3] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_F:
        data.ctrl[3] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_G:
        data.ctrl[6] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_H:
        data.ctrl[6] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_J:
        data.ctrl[9] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_K:
        data.ctrl[9] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_L:
        data.ctrl[13] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_SEMICOLON:
        data.ctrl[13] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_Z:
        data.ctrl[19] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_X:
        data.ctrl[19] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_C:
        data.ctrl[4] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_V:
        data.ctrl[4] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_B:
        data.ctrl[7] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_N:
        data.ctrl[7] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_M:
        data.ctrl[10] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_COMMA:
        data.ctrl[10] -= 0.1

    if act == glfw.PRESS and key == glfw.KEY_PERIOD:
        data.ctrl[14] += 0.1

    if act == glfw.PRESS and key == glfw.KEY_SLASH:
        data.ctrl[14] -= 0.1

    # if act == glfw.PRESS and key == glfw.KEY_C:
    #     data.ctrl[19] += 0.1
    #
    # if act == glfw.PRESS and key == glfw.KEY_V:
    #     data.ctrl[18] -= 0.1
    #
    # if act == glfw.PRESS and key == glfw.KEY_B:
    #     data.ctrl[19] += 0.1
    #
    # if act == glfw.PRESS and key == glfw.KEY_N:
    #     data.ctrl[19] -= 0.1


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
    # data.ctrl[:23] = gdf_d
    counter = data.time
    while not glfw.window_should_close(window):
        time_prev = data.time

        while data.time - time_prev < 1.0 / 60.0:
            mj.mj_step(model, data)

        if data.time - counter > 0.1:
            counter = data.time

            if data.site_xpos[20][2] < 1:
                sample = synth(frequencies_dict["A0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[21][2] < 1:
                sample = synth(frequencies_dict["A#0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[22][2] < 1:
                sample = synth(frequencies_dict["B0"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[23][2] < 1:
                sample = synth(frequencies_dict["C1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[24][2] < 1:
                sample = synth(frequencies_dict["C#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[25][2] < 1:
                sample = synth(frequencies_dict["D1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[26][2] < 1:
                sample = synth(frequencies_dict["D#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[27][2] < 1:
                sample = synth(frequencies_dict["E1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[28][2] < 1:
                sample = synth(frequencies_dict["F1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[29][2] < 1:
                sample = synth(frequencies_dict["F#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[30][2] < 1:
                sample = synth(frequencies_dict["G1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[31][2] < 1:
                sample = synth(frequencies_dict["G#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[32][2] < 1:
                sample = synth(frequencies_dict["A1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[33][2] < 1:
                sample = synth(frequencies_dict["A#1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[34][2] < 1:
                sample = synth(frequencies_dict["B1"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[35][2] < 1:
                sample = synth(frequencies_dict["C2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[36][2] < 1:
                sample = synth(frequencies_dict["C#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[37][2] < 1:
                sample = synth(frequencies_dict["D2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[38][2] < 1:
                sample = synth(frequencies_dict["D#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[39][2] < 1:
                sample = synth(frequencies_dict["E2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[40][2] < 1:
                sample = synth(frequencies_dict["F2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[41][2] < 1:
                sample = synth(frequencies_dict["F#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[42][2] < 1:
                sample = synth(frequencies_dict["G2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[43][2] < 1:
                sample = synth(frequencies_dict["G#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[44][2] < 1:
                sample = synth(frequencies_dict["A2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[45][2] < 1:
                sample = synth(frequencies_dict["A#2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[46][2] < 1:
                sample = synth(frequencies_dict["B2"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[47][2] < 1:
                sample = synth(frequencies_dict["C3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[48][2] < 1:
                sample = synth(frequencies_dict["C#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[49][2] < 1:
                sample = synth(frequencies_dict["D3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[50][2] < 1:
                sample = synth(frequencies_dict["D#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[51][2] < 1:
                sample = synth(frequencies_dict["E3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[52][2] < 1:
                sample = synth(frequencies_dict["F3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[53][2] < 1:
                sample = synth(frequencies_dict["F#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[54][2] < 1:
                sample = synth(frequencies_dict["G3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[55][2] < 1:
                sample = synth(frequencies_dict["G#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[56][2] < 1:
                sample = synth(frequencies_dict["A3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[57][2] < 1:
                sample = synth(frequencies_dict["A#3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[58][2] < 1:
                sample = synth(frequencies_dict["B3"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[59][2] < 1:
                sample = synth(frequencies_dict["C4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[60][2] < 1:
                sample = synth(frequencies_dict["C#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[61][2] < 1:
                sample = synth(frequencies_dict["D4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[62][2] < 1:
                sample = synth(frequencies_dict["D#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[63][2] < 1:
                sample = synth(frequencies_dict["E4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[64][2] < 1:
                sample = synth(frequencies_dict["F4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[65][2] < 1:
                sample = synth(frequencies_dict["F#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[66][2] < 1:
                sample = synth(frequencies_dict["G4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[67][2] < 1:
                sample = synth(frequencies_dict["G#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[68][2] < 1:
                sample = synth(frequencies_dict["A4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[69][2] < 1:
                sample = synth(frequencies_dict["A#4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[70][2] < 1:
                sample = synth(frequencies_dict["B4"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[71][2] < 1:
                sample = synth(frequencies_dict["C5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[72][2] < 1:
                sample = synth(frequencies_dict["C#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[73][2] < 1:
                sample = synth(frequencies_dict["D5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[74][2] < 1:
                sample = synth(frequencies_dict["D#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[75][2] < 1:
                sample = synth(frequencies_dict["E5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[76][2] < 1:
                sample = synth(frequencies_dict["F5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[77][2] < 1:
                sample = synth(frequencies_dict["F#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[78][2] < 1:
                sample = synth(frequencies_dict["G5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[79][2] < 1:
                sample = synth(frequencies_dict["G#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[80][2] < 1:
                sample = synth(frequencies_dict["A5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[81][2] < 1:
                sample = synth(frequencies_dict["A#5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[82][2] < 1:
                sample = synth(frequencies_dict["B5"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[83][2] < 1:
                sample = synth(frequencies_dict["C6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[84][2] < 1:
                sample = synth(frequencies_dict["C#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[85][2] < 1:
                sample = synth(frequencies_dict["D6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[86][2] < 1:
                sample = synth(frequencies_dict["D#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[87][2] < 1:
                sample = synth(frequencies_dict["E6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[88][2] < 1:
                sample = synth(frequencies_dict["F6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[89][2] < 1:
                sample = synth(frequencies_dict["F#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[90][2] < 1:
                sample = synth(frequencies_dict["G6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[91][2] < 1:
                sample = synth(frequencies_dict["G#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[92][2] < 1:
                sample = synth(frequencies_dict["A6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[93][2] < 1:
                sample = synth(frequencies_dict["A#6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[94][2] < 1:
                sample = synth(frequencies_dict["B6"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[95][2] < 1:
                sample = synth(frequencies_dict["C7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[96][2] < 1:
                sample = synth(frequencies_dict["C#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[97][2] < 1:
                sample = synth(frequencies_dict["D7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[98][2] < 1:
                sample = synth(frequencies_dict["D#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[99][2] < 1:
                sample = synth(frequencies_dict["E7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[100][2] < 1:
                sample = synth(frequencies_dict["F7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[101][2] < 1:
                sample = synth(frequencies_dict["F#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[102][2] < 1:
                sample = synth(frequencies_dict["G7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[103][2] < 1:
                sample = synth(frequencies_dict["G#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[104][2] < 1:
                sample = synth(frequencies_dict["A7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[105][2] < 1:
                sample = synth(frequencies_dict["A#7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)
            if data.site_xpos[106][2] < 1:
                sample = synth(frequencies_dict["B7"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

            if data.site_xpos[107][2] < 1:
                sample = synth(frequencies_dict["C8"])
                sample.set_volume(0.1)
                sample.play()
                sample.fadeout(1_000)

        # x y z position of the free joint
        # print(data.qpos[0])
        # print(data.qpos[1])
        # print(data.qpos[2])

        # quat = np.array([data.qpos[3],data.qpos[4],data.qpos[5],data.qpos[6]])
        # euler = quat2euler(quat)
        # print('yaw = ',euler[2]);

        # print(data.site_xpos[0])

        # if data.time>=simend:
        #     break

        # get framebuffer viewport

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


free = np.asarray([0., 0., 0., 0.1, 0.5, 0., 0., 0., 0., 0., 0., 0.,
                   0., 0., 0., 0., 0., 0., 0., 0., 0.185, 0.08, -0.01, ])
pressed = np.asarray([0., 0., 0., 0.4, 0.5, 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0.185, 0.08, -0.01, ])
# f = free, p = pressed, chords section
adfa2_f = np.asarray([-0.4, -0.2, 0.3, 0.4, 0.5, 0.2, 0.2, 0.4, 0.2, 0.5, 0.3, 0.,
                      -0.3, 0.5, 0.3, 0., 0.4, 0., 0.3, -0.4, 0.25, 0.1, -0.03, ])
adfa2_p = np.asarray([-0.4, -0.2, 0.3, 0.4, 0.5, 0.2, 0.2, 0.4, 0.2, 0.5, 0.3, 0.,
                      -0.3, 0.5, 0.3, 0., 0.4, 0., 0.3, -0.4, 0.25, 0.1, -0.06, ])
cege3_f = np.asarray([-2.00000000e-01, -4.00000000e-01, 4.00000000e-01, 2.00000000e-01,
                      7.00000000e-01, 3.00000000e-01, -2.77555756e-17, 4.00000000e-01,
                      3.00000000e-01, 3.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                      -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, 0.00000000e+00,
                      2.00000000e-01, 0.00000000e+00, 1.00000000e-01, -5.00000000e-01,
                      1.80000000e-01, 1.00000000e-01, -6.00000000e-02, ])
cege3_p = np.asarray([-2.00000000e-01, -4.00000000e-01, 4.00000000e-01, 2.00000000e-01,
                      7.00000000e-01, 3.00000000e-01, -2.77555756e-17, 4.00000000e-01,
                      3.00000000e-01, 3.00000000e-01, 5.00000000e-01, 0.00000000e+00,
                      -1.00000000e-01, 2.00000000e-01, 6.00000000e-01, 0.00000000e+00,
                      2.00000000e-01, 0.00000000e+00, 1.00000000e-01, -5.00000000e-01,
                      1.80000000e-01, 1.00000000e-01, -1.10000000e-01, ])

beat = 0.375  # 160 BPM / 8%3 BPS / 0.375 SPB (B=Beats, P=Per, M=Minutes, S=Seconds)

whole = 4 * beat
half = 2 * beat
quarter = beat
eighth = beat / 2
sixteenth = beat / 4


async def sequence():
    await asyncio.sleep(1)
    data.ctrl[:23] = free
    await asyncio.sleep(1)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(quarter)

    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = free
    await asyncio.sleep(eighth)
    data.ctrl[:23] = pressed
    await asyncio.sleep(eighth)
    data.ctrl[:23] = adfa2_f
    await asyncio.sleep(quarter)

    data.ctrl[:23] = adfa2_p
    await asyncio.sleep(quarter)
    data.ctrl[:23] = cege3_f
    await asyncio.sleep(quarter)
    data.ctrl[:23] = cege3_p
    await asyncio.sleep(quarter)
    data.ctrl[:23] = cege3_f
    await asyncio.sleep(quarter)


async def main():
    task1 = asyncio.create_task(side())
    task2 = asyncio.create_task(sequence())

    await task2
    await task1


asyncio.run(main())

glfw.terminate()
