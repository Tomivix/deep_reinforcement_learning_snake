import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

import os
import DQNAgent
#from DQNAgent import MODEL_NAME

#...
# issue: https://stackoverflow.com/questions/56961856/how-to-write-to-tensorboard-in-tensorflow-2/56961915

# Own Tensorboard class
# class ModifiedTensorBoard(TensorBoard):
#
#     # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.step = 1
#         self.writer = tf.summary.FileWriter(self.log_dir) # tu sie moze wywalac skubaniec wiec lepiej nie uzywac tej klasy jeszcze
#
#     # Overriding this method to stop creating default log writer
#     def set_model(self, model):
#         pass
#
#     # Overrided, saves logs with our step number
#     # (otherwise every .fit() will start writing from 0th step)
#     def on_epoch_end(self, epoch, logs=None):
#         self.update_stats(**logs)
#
#     # Overrided
#     # We train for one batch only, no need to save anything at epoch end
#     def on_batch_end(self, batch, logs=None):
#         pass
#
#     # Overrided, so won't close writer
#     def on_train_end(self, _):
#         pass
#
#     # Custom method for saving own metrics
#     # Creates writer, writes custom metrics and closes writer
#     def update_stats(self, **stats):
#         self._write_logs(stats, self.step)


class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, DQNAgent.MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    def on_train_batch_end(self, batch, logs=None):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        self._write_logs(stats, self.step)

    # Added because of version
    def _write_logs(self, logs, index):
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=index)
                self.step += 1
                self.writer.flush()