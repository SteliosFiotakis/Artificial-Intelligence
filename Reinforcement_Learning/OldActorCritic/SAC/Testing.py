# import gym
# # from gym.wrappers import RescaleAction
# #
# # env = gym.make("InvertedPendulum-v4")
# #
# # print(env.action_space)
# #
# # env = RescaleAction(env, min_action=-1, max_action=1)
# #
# # print(env.action_space)
#
# env = gym.make("BipedalWalker-v3")
#
# print(env.observation_space.shape)
# print(env.action_space.shape)
#
# import math
#
# # Define the set of numbers
# numbers = [3, 4, 6]
#
# # Compute the product of the numbers
# product = 1
# for number in numbers:
#     product *= number
#
# # Compute the GCD of the numbers
# gcd = math.gcd(numbers[0], numbers[1])
# for number in numbers[2:]:
#     gcd = math.gcd(gcd, number)
#
# # Compute and print the LCM
# lcm = product // gcd
# print(lcm)

variable = 34
variable2 = 3
print(type(variable))

print(tuple([variable, variable2]))
print(tuple([variable + variable2]))t

