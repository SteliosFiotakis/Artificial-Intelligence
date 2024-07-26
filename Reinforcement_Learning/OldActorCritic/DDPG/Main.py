import gym
import numpy as np
from Agent import Agent


##################################################
# HYPERPARAMETERS
##################################################

env_id = "Hopper-v4"
model_saving_path = f"./Saves/{env_id}/"
learning_plot_saving_path = "Plots/"
replay_buffer_size = 1_000_000
total_steps = 5_000_000
episode_steps = 500
batch_size = 32
gamma = 0.99
load = False
tau = 5e-03
actor_structure = [512, 512]
critic_structure = [512, 512]
alpha = 1e-03
beta = 2e-03

##################################################
# TRAINING
##################################################

episode = 0
global_step = 0
env = gym.make(env_id, terminate_when_unhealthy=False)
best_return = env.reward_range[0]
average_returns = list()
total_returns = list()
agent = Agent(critic_architecture=critic_structure,
              actor_architecture=actor_structure,
              replay_buffer_s=replay_buffer_size,
              model_path=model_saving_path,
              batch_size=batch_size,
              environment=env,
              do_load=load,
              alpha=alpha,
              gamma=gamma,
              beta=beta,
              tau=tau)

state = env.reset()[0]
for temp_step in range(batch_size):
    action = env.action_space.sample()
    next_state, reward, terminated, truncated, info = env.step(action)
    done = max(terminated, truncated)
    agent.remember(state, action, reward, next_state, done)

while global_step < total_steps:
    state = env.reset()[0]
    episode_return = 0

    for step in range(1, episode_steps+1):
        global_step += 1

        action = agent.choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = max(terminated, truncated)
        agent.remember(state, action, reward, next_state, done)
        agent.learn()

        state = next_state
        episode_return += reward
        if done:
            break
    total_returns.append(episode_return)
    average_returns = np.mean(total_returns[-100:])

    if average_returns > best_return:
        best_return = average_returns
        agent.save()

    print(f"Episode: {episode}, Reward: {episode_return}, Steps: {step}, Global step: {global_step:_}")
    episode += 1
