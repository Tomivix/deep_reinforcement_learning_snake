import gym
import gym_snake
import numpy as np
import tensorflow as tf

import time
import os
import random
from tqdm import tqdm
# from DQNAgent import DQNAgent, MODEL_NAME  # comment for convnets
from DQNAgentSimpleNN import DQNAgentSimpleNN, MODEL_NAME
from env_converter import get_input_for_nn


# Environment settings
EPISODES = 60_000

# Exploration settings
START_EPSILON = 0.2
epsilon = START_EPSILON  # not a constant, going to be decayed ## I changed from  1 to 0.1

EPSILON_DECAY = 0.999
MIN_EPSILON = 0.001
min_epsilon_counter = 0

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
env.unit_size = 1
env.unit_gap = 0


## Observing snake for now
obs = env.reset()

# agent = DQNAgent(obs.shape) # for convnets
agent = DQNAgentSimpleNN((28,))  # for simple nn

# Controller
game_controller = env.controller

# Grid
grid_object = game_controller.grid
grid_pixels = grid_object.grid

# Snake(s)
snakes_array = game_controller.snakes
snake_object1 = snakes_array[0]

# For stats
ep_rewards = [-1]
apple_rewards = [0]

MIN_REWARD = 1

print(agent.model.summary())

# Iterate over episodes
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit='episodes'):

    #env.render()  # Render latest instance of game

    # Update tensorboard step every episode
    agent.tensorboard.step = episode

    # Restarting episode - reset episode reward and step number
    episode_reward = 0
    step = 1

    # Reset environment and get initial state
    current_state = env.reset()
    current_state = get_input_for_nn(env, 0)  # comment for convnets

    # Reset flag and start iterating until episode ends
    done = False
    apple_count = 0
    while not done:
        # env.render()  # for testing

        # if env.controller.snakes[0] is not None:
        #    nn_input = get_input_for_nn(env, 0)
        #    for i in range(0, 28):
        #        print(f'{episode} {step} {i} {nn_input[i]}')
        #    print('==========================')

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            #action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            action = np.random.randint(0, 3)

        new_state, reward, done, info = env.step([action])
        new_state = get_input_for_nn(env, 0)  # comment line for convnets

        if reward == 50:
            apple_count += 1

        # Transform new continuous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

        if step > 300:
            done = True

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    apple_rewards.append(apple_count)
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])

        average_apple = sum(apple_rewards[-AGGREGATE_STATS_EVERY:])/len(apple_rewards[-AGGREGATE_STATS_EVERY:])
        min_apple = min(apple_rewards[-AGGREGATE_STATS_EVERY:])
        max_apple = max(apple_rewards[-AGGREGATE_STATS_EVERY:])

        print()
        print('==========================')
        print(f'episode: {episode}')
        print(f'average_reward: {average_reward}')
        print(f'min_reward: {min_reward}')
        print(f'max_reward: {max_reward}')
        print(f'epsilon: {epsilon}')
        print()
        print(f'average_apple: {average_apple}')
        print(f'min_apple: {min_apple}')
        print(f'max_apple: {max_apple}')
        print('==========================')
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon, average_apple=average_apple, min_apple=min_apple, max_apple=max_apple)
        #agent.tensorboard._write_logs(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model every 500 episodes
        #if min_reward >= MIN_REWARD:
        if episode % 500 == 0 or episode == 1:
            #agent.model.save(f'models\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            agent.model.save(f'models\{MODEL_NAME}_{episode}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
    else:
        min_epsilon_counter += 1
        if min_epsilon_counter % 500 == 0:
            epsilon = START_EPSILON

env.close()




