#!/usr/bin/env python3

import time
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import PPO
from stable_baselines3.td3 import MlpPolicy as td3ppoMlpPolicy
from stable_baselines3.common.policies import ActorCriticPolicy as ACP
from stable_baselines3.common.monitor import Monitor

from HA import HoverAviary

env = HoverAviary(gui = False, record = False, freq = 100)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

env = Monitor(env)

#model = PPO(ACP, env, verbose = 1)
model = PPO.load("HA_PPOagent_1.zip", env = env)

n_ep = 100
model.learn(52*n_ep, eval_freq = 2)

new_save = False
if new_save:
	model.save("HA_PPOagent_2")
else:
	model.save("HA_PPOagent_1")

print(env.get_episode_rewards())

plt.plot([i for i in range(len(env.get_episode_rewards()))],env.get_episode_rewards())
plt.show()

env = HoverAviary(gui = True, record = False, freq = 50)

obs = env.reset()
rew = []

for i in range(10):
	done = False
	env.reset()
	tot = 0
	step = 1
	while not done:
		action, _state = model.predict(obs)
		obs, reward, done, _= env.step(action)
		tot += reward
		print(step)
		step += 1
	rew.append(tot)
print(rew)
env.close()