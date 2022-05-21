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
from gym_pybullet_drones.utils.Logger import Logger

from HA import HoverAviary

env = HoverAviary(gui = False, record = False, freq = 100)

print("[INFO] Action space:", env.action_space)
print("[INFO] Observation space:", env.observation_space)

env = Monitor(env)

policy_kwargs = dict(net_arch=[512,512,256,128])
model = PPO(ACP, env, verbose = 1, policy_kwargs = policy_kwargs)
#model = PPO(ACP, env, verbose = 1)
#model = PPO.load("HA_PPOagent_1L.zip", env = env)
#model = PPO.load("HA_PPOagent_3L.zip", env = env)
#model = PPO.load("HA_PPOagent_4L.zip", env = env)
#model = PPO.load("ppo_hover_2503_01.zip", env = env)

n_ep = 1000

model.learn(3e6, eval_freq = 100)
model.save("HA_PPOagent_2105_1")

print(env.get_episode_rewards())

plt.plot([i for i in range(len(env.get_episode_rewards()))],env.get_episode_rewards())
plt.show()

logger = Logger(logging_freq_hz=int(1),
                num_drones=1)

env = HoverAviary(gui = True, record = False, freq = 100)

obs = env.reset()
rew = []

for i in range(2):
	done = False
	env.reset()
	tot = 0
	step = 1
	while not done:
		action, _state = model.predict(obs)
		obs, reward, done, _= env.step(action)
		tot += reward
		#print(step)
		step += 1
		logger.log(drone=0,
                   timestamp=i/env.SIM_FREQ,
                   state=np.hstack([obs[0:3], np.zeros(4), obs[3:15], np.resize(action, (4))]),
                   control=np.zeros(12))
	rew.append(tot)
	print(step, i)
print(rew)
env.close()
logger.plot()