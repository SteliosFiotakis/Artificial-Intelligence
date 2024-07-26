# import gym
#
# env = gym.make("Humanoid-v4")

episode = [i for i in range(1_000, 4_000)]
score_history = [i for i in range(4_000, 7_000)]
with open("test.txt", "w") as file:
    file.write(str(episode))
    file.write("\n")
    file.write(str(score_history))

with open("test.txt", "r") as file:
    data1 = eval(file.readline())
    data2 = eval(file.readline())

print(data1)
print(data2)
