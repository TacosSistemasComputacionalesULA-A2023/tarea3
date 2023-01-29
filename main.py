import os

import gym
import gym_taco_environments
import time
from agent import ValueIteration
import numpy as np

# Allowing environment to have sounds
if "SDL_AUDIODRIVER" in os.environ:
    del os.environ["SDL_AUDIODRIVER"]

env = gym.make('FrozenLake-v0', render_mode='ansi', delay=0)
space_num = env.observation_space.n
agent = ValueIteration(
    states_n=space_num,
    actions_n=env.action_space.n,
    P=env.P, gamma=0.09,
)

agent.solve(
    policy_evaluations=100000, 
    iterations=100000, 
    delta=0.09, method='politer',
)

agent.render()
observation, info = env.reset()
terminated, truncated = False, False

env.render()

while not (terminated or truncated):
    action = agent.get_action(observation)
    observation, reward, terminated, truncated, _ = env.step(action)

if reward == 1.0:
    print('gift reached')

env.close()
