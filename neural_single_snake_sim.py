import gym
import gym_snake
#from DQNAgent import DQNAgent
import numpy as np
import tensorflow as tf

import time
import os
import random

DISCOUNT = 0.97
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
#MIN_REWARD = -200  # For model save
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
#if not os.path.isdir('models'):
#    os.makedirs('models')

# same values when calling random

#random.seed(1)
#np.random.seed(1)
#tf.random.set_seed(1)



## Creating environment
env = gym.make('snake-v0')
env.grid_size = [10, 10]

## Observing snake for now
obs = env.reset()

import tensorflow as tf

from tensorflow.compat.v1.keras.backend import set_session

config = tf.compat.v1.ConfigProto()

config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

config.log_device_placement = True  # to log device placement (on which device the operation ran)

sess = tf.compat.v1.Session(config=config)

#agent = DQNAgent(obs.shape)
# unikanie scian 8_16_f_16d_2500.model'
model = tf.keras.models.load_model('models\8_16_f_16d_2500.model')

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]



set_session(sess)

for i in range(100):
	obs = env.reset()
	done = False
	
	no_of_moves = 0
	while not done:  # run for 1000 steps
		env.render()  # Render latest instance of game
		action = env.action_space.sample()  # Random action
		action = np.argmax(model.predict(obs.reshape(-1, *obs.shape)/255)[0])
		#action = model.predict
		print('action -> ', action)
		print(f'EPISODE:  {i}')
		obs, rewards, done, info = env.step([action])  # Implement action
		
		no_of_moves += 1
		
		#print('last_obs', last_obs)
		#print('last_obs.shape', last_obs.shape)
		#print('reward', rewards)
		#print('done', done)
		#print('info', info)
		
		print('===================')
		#time.sleep(10)
		#if done:
		#	env.reset()
		
		if no_of_moves > 150:
			done = True
env.close()