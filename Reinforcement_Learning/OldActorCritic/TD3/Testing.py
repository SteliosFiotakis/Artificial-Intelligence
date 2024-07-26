# import gym
#
# env_id = "Walker2d-v4"
#
# env = gym.make(env_id, render_mode='human')
#
# while True:
#     env.reset()
#     done = False
#
#     while not done:
#         action = env.action_space.sample()
#         _, _, terminated, truncated, _ = env.step(action)

# user_input = input("Give input please: ")
#
# print(type(eval(user_input)))

from ActorCriticFamily.common.Utils import save_learning_plot
import random

x = [i + 1 for i in range(1_000)]
scores = [random.randint(-100, 100) for i in range(1_000)]
filename = "Plots/Hi"
save_learning_plot(x, scores, filename)
