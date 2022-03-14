#!/usr/bin/env python3

import time
import gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3 import DDPG
from stable_baselines3.td3 import MlpPolicy as td3ddpgMlpPolicy
from gym_pybullet_drones.envs.single_agent_rl.HoverAviary import HoverAviary

EPISODE_REWARD_THRESHOLD = -0

env = HoverAviary(gui = False, record = False)
print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)
model = DDPG(td3ddpgMlpPolicy, env, verbose = 1)
model.learn(1200*10, eval_freq = 2)

env = HoverAviary(gui = True, record = False)
obs = env.reset()
rew = []

for i in range(10):
	done = False
	env.reset()
	tot = 0
	while not done:
		action, _state = model.predict(obs, deterministic = True)
		obs, reward, done, _= env.step(action)
		tot += reward
		if done:
			obs = env.reset()
			rew.append(tot)