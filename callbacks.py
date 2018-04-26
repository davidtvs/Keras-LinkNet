import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import utils


class TensorBoardPrediction(Callback):
    def __init__(self, generator, class_to_rgb, log_dir, batch_index=0):
        super().__init__()

        self.generator = generator
        self.class_to_rgb = class_to_rgb
        self.batch_index = batch_index
        self.log_dir = log_dir
        self.sess = None
        self.summary_op = None

    def on_epoch_end(self, epoch, logs=None):
        sample, y_true = self.generator[self.batch_index]
        y_pred = np.asarray(self.model.predict_on_batch(sample))

        # Convert y_true and y_pred from categorical to RGB images
        y_true = utils.categorical_to_rgb(y_true, self.class_to_rgb)
        y_pred = utils.categorical_to_rgb(y_pred, self.class_to_rgb)

        if self.sess is None:
            self.sess = tf.keras.backend.get_session()

            # Create a summary to monitor the sample, target and prediction
            tf.summary.image('samples', tf.convert_to_tensor(sample))
            tf.summary.image('targets', tf.convert_to_tensor(y_true))
            tf.summary.image('predictions', tf.convert_to_tensor(y_pred))
            self.summary_op = tf.summary.merge_all()

        # Add the summaries; the summary op must be evaluated first to get the
        # tf.Summary protocol buffer
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(self.summary_op.eval(session=self.sess), epoch)
        writer.close()
