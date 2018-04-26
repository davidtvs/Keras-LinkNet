import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import utils


class TensorBoardPrediction(Callback):
    """A TensorBoard callback to display samples, targets, and predictions.

    Args:
        generator (keras.utils.Sequence): A data generator to iterate over the
            dataset.
        class_to_rgb (OrderedDict): An ordered dictionary that relates pixel
            values, class names, and class colors.
        log_dir (string): Specifies the directory where TensorBoard will
            write TensorFlow event files that it can display.
        batch_index (int): The batch index to display. Default: 0.
        max_outputs (int): Max number of elements in the batch to generate
            images for. Default: 3.

    """

    def __init__(
        self, generator, class_to_rgb, log_dir, batch_index=0, max_outputs=3
    ):
        super().__init__()

        self.generator = generator
        self.class_to_rgb = class_to_rgb
        self.batch_index = batch_index
        self.log_dir = log_dir
        self.max_outputs = max_outputs
        self.sess = None
        self.summary_op = None

    def on_epoch_end(self, epoch, logs=None):
        """Creates and updates the event files.

        Args:
            epoch (int): Current epoch.
            logs (dict): Includes training accuracy and loss, and, if
                validation is enabled, validation accuracy and loss.
                Default: None.

        """
        sample, y_true = self.generator[self.batch_index]
        y_pred = np.asarray(self.model.predict_on_batch(sample))

        # Convert y_true and y_pred from categorical to RGB images
        y_true = utils.categorical_to_rgb(y_true, self.class_to_rgb)
        y_pred = utils.categorical_to_rgb(y_pred, self.class_to_rgb)

        if self.sess is None:
            self.sess = tf.keras.backend.get_session()

            # Create a summary to monitor the sample, target and prediction
            tf.summary.image(
                'samples',
                tf.convert_to_tensor(sample),
                max_outputs=self.max_outputs
            )
            tf.summary.image(
                'targets',
                tf.convert_to_tensor(y_true),
                max_outputs=self.max_outputs
            )
            tf.summary.image(
                'predictions',
                tf.convert_to_tensor(y_pred),
                max_outputs=self.max_outputs
            )
            self.summary_op = tf.summary.merge_all()

        # Add the summaries; the summary op must be evaluated first to get the
        # tf.Summary protocol buffer
        writer = tf.summary.FileWriter(self.log_dir)
        writer.add_summary(self.summary_op.eval(session=self.sess), epoch)
        writer.close()
