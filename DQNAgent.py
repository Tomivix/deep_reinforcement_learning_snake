import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Conv2D, Dense, Dropout, Flatten, MaxPool2D
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam

from collections import deque

print(tf.test.is_gpu_available())

DISCOUNT = 0.97
REPLAY_MEMORY_SIZE = 50_000 # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)

MODEL_NAME = "8_16_f_16d"

class DQNAgent:
    def __init__(self, env_shape):

        # Main model
        self.model = self.create_model(env_shape)

        # target network
        self.target_model = self.create_model(env_shape)
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard= tf.keras.callbacks.TensorBoard(log_dir='logs\\')


    def create_model(self, env_shape):
        model = Sequential()

        model.add(Conv2D(8, (3, 3), activation='relu', input_shape=env_shape))
        model.add(Dropout(0.2))

        model.add(Conv2D(16, (3, 3), activation='relu'))
        model.add(MaxPool2D((5, 5)))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(16))
        model.add(Dense(4, activation='linear')) # 4 = action space size

        model.compile(loss="mse", optimizer=Adam(lr=0.001), metrics=['accuracy'])
        print(model.summary())
        return model
