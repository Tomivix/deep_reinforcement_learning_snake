import gym
import gym_snake
import numpy as np
import tensorflow as tf

import time
import os
import random
from tqdm import tqdm
from DQNAgent import DQNAgent, MODEL_NAME

MEMORY_FRACTION = 0.20 # useful to train multiple snakes


# Environment settings
EPISODES = 20_000

# Exploration settings
epsilon = 0.5  # not a constant, going to be decayed #### zmienilem z 1 na 0.1
EPSILON_DECAY = 0.99965
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
env.unit_size = 1
env.unit_gap = 0

## Observing snake for now
obs = env.reset()

agent = DQNAgent(obs.shape)

# Definitions of values at particular positions:
#  0 - (int) distance between snake's head and the top wall
#  1 - (int) distance between snake's head and the right wall
#  2 - (int) distance between snake's head and the bottom wall
#  3 - (int) distance between snake's head and the left wall
#  4 - (bool) can snake see their body to the north?
#  5 - (bool) can snake see their body to the northeast?
#  6 - (bool) can snake see their body to the east?
#  7 - (bool) can snake see their body to the southeast?
#  8 - (bool) can snake see their body to the south?
#  9 - (bool) can snake see their body to the southwest?
# 10 - (bool) can snake see their body to the west?
# 11 - (bool) can snake see their body to the northwest?
# 12 - (bool) can snake see food to the north?
# 13 - (bool) can snake see food to the northeast?
# 14 - (bool) can snake see food to the east?
# 15 - (bool) can snake see food to the southeast?
# 16 - (bool) can snake see food to the south?
# 17 - (bool) can snake see food to the southwest?
# 18 - (bool) can snake see food to the west?
# 19 - (bool) can snake see food to the northwest?
# 20 - (int) is snake's head moving to the north?
# 21 - (int) is snake's head moving to the east?
# 22 - (int) is snake's head moving to the south?
# 23 - (int) is snake's head moving to the west?
# 24 - (int) is snake's tail moving to the north?
# 25 - (int) is snake's tail moving to the east?
# 26 - (int) is snake's tail moving to the south?
# 27 - (int) is snake's tail moving to the west?


def get_input_for_nn(envir, id):
    values = [None] * 28

    snake = envir.controller.snakes[id]

    sx = snake.head[0]
    sy = snake.head[1]
    mx = envir.grid_size[0] - 1
    my = envir.grid_size[1] - 1

    bc = envir.controller.grid.BODY_COLOR
    fc = envir.controller.grid.FOOD_COLOR

    values[0] = sy
    values[1] = mx - sx
    values[2] = my - sy
    values[3] = sx

    for ind in range(4, 20):
        values[ind] = False

    for cy in range(sy - 1, -1, -1):
        color = envir.controller.grid.color_of((sx, cy))
        if np.array_equal(color, bc):
            values[4] = True
        elif np.array_equal(color, fc):
            values[12] = True

    for cx in range(sx + 1, mx + 1):
        color = envir.controller.grid.color_of((cx, sy))
        if np.array_equal(color, bc):
            values[6] = True
        elif np.array_equal(color, fc):
            values[14] = True

    for cy in range(sy + 1, my + 1):
        color = envir.controller.grid.color_of((sx, cy))
        if np.array_equal(color, bc):
            values[8] = True
        elif np.array_equal(color, fc):
            values[16] = True

    for cx in range(sx - 1, -1, -1):
        color = envir.controller.grid.color_of((cx, sy))
        if np.array_equal(color, bc):
            values[10] = True
        elif np.array_equal(color, fc):
            values[18] = True

    md = min(sx, my - sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx - cd, sy + cd))
        if np.array_equal(color, bc):
            values[5] = True
        elif np.array_equal(color, fc):
            values[13] = True

    md = min(mx - sx, my - sy)
    for cd in range(0, md + 1) :
        color = envir.controller.grid.color_of((sx + cd, sy + cd))
        if np.array_equal(color, bc):
            values[7] = True
        elif np.array_equal(color, fc):
            values[15] = True

    md = min(mx - sx, sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx + cd, sy - cd))
        if np.array_equal(color, bc):
            values[9] = True
        elif np.array_equal(color, fc):
            values[17] = True

    md = min(sx, sy)
    for cd in range(0, md + 1):
        color = envir.controller.grid.color_of((sx - cd, sy - cd))
        if np.array_equal(color, bc):
            values[11] = True
        elif np.array_equal(color, fc):
            values[19] = True

    for i in range(0, 4):
        values[20 + i] = 1 if snake.direction == i else 0

    tx = snake.body[0][0]
    ty = snake.body[0][1]
    sx = snake.body[1][0]
    sy = snake.body[1][1]

    values[24] = 1 if ty - sy == 1 else 0
    values[25] = 1 if sx - tx == 1 else 0
    values[26] = 1 if sy - ty == 1 else 0
    values[27] = 1 if tx - sx == 1 else 0

    return values


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

    # Reset flag and start iterating until episode ends
    done = False
    apple_count = 0
    while not done:

        #env.render() podczas testowania

        if env.controller.snakes[0] is not None:
            nn_input = get_input_for_nn(env, 0)
            for i in range(0, 28):
                print(f'{episode} {step} {i} {nn_input[i]}')
            print('==========================')

        # This part stays mostly the same, the change is to query a model for Q values
        if np.random.random() > epsilon:
            # Get action from Q table
            action = np.argmax(agent.get_qs(current_state))
        else:
            # Get random action
            #action = np.random.randint(0, env.ACTION_SPACE_SIZE)
            action = np.random.randint(0, 3)

        # print('action -> ', action)
        # last_obs, rewards, done, info = env.step(action)  # Implement action
        # print('last_obs', last_obs)
        # print('last_obs.shape', last_obs.shape)
        # print('reward', rewards)
        # print('done', done)
        # print('info', info)
        # print('===================')
        # time.sleep(10)
        # if done:
        #     env.reset()

        new_state, reward, done, info = env.step([action])

        if reward == 50:
            apple_count += 1

        # Transform new continous state to new discrete state and count reward
        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Every step we update replay memory and train main network
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

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
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon) # tak bylo
        #agent.tensorboard._write_logs(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Save model, but only when min reward is greater or equal a set value
        #if min_reward >= MIN_REWARD:
        if episode % 500 == 0 or episode == 1:
            #agent.model.save(f'models\{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')
            agent.model.save(f'models\{MODEL_NAME}_{episode}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)

env.close()




