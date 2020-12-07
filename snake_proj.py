import gym
import gym_snake
from DQNAgent import DQNAgent
import numpy as np
import tensorflow as tf

import time
import os
import random

MEMORY_FRACTION = 0.20 # useful to train multiple snakes


# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = False

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# same values when calling random
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)



## Creating environment
env = gym.make('snake-v0')
env.grid_size = [10, 10]

## Observing snake for now
obs = env.reset()

agent = DQNAgent(obs.shape)

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]



done = False
for _ in range(1000):  # run for 1000 steps
    env.render()  # Render latest instance of game
    action = env.action_space.sample()  # Random action
    print('action -> ', action)

    #print('action -> ', action)
    last_obs, rewards, done, info = env.step(action)  # Implement action
    print('last_obs', last_obs)
    print('last_obs.shape', last_obs.shape)
    print('reward', rewards)
    print('done', done)
    print('info', info)
    print('===================')
    #time.sleep(10)
    if done:
        env.reset()
env.close()