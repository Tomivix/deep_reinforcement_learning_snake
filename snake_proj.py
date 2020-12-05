# from tensorflow.keras.layers import Flatten
# print(Flatten.__doc__)

import gym
import gym_snake


## Creating environment
env = gym.make('snake-v0')
obs = env.reset()

## Observing snake for now
obs = env.reset()

for _ in range(100):  # run for 1000 steps
    env.render()  # Render latest instance of game
    action = env.action_space.sample()  # Random action
    env.step(action)  # Implement action

env.close()