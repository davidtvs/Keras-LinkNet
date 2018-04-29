from io import BytesIO
import numpy as np
import tensorflow as tf
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from data.utils import categorical_to_rgb


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
        self.max_outputs = max_outputs
        self.log_dir = log_dir

    def set_model(self, model):
        super().set_model(model)
        self.writer = tf.summary.FileWriter(self.log_dir)

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
        y_true = categorical_to_rgb(y_true, self.class_to_rgb)
        y_pred = categorical_to_rgb(y_pred, self.class_to_rgb)

        batch_summary = self.image_summary(sample, 'sample')
        batch_summary += self.image_summary(y_true, 'target')
        batch_summary += self.image_summary(y_pred, 'prediction')
        summary = tf.Summary(value=batch_summary)

        # Write the summaries to the file
        self.writer.add_summary(summary, epoch)
        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()

    def image_summary(self, batch, tag):
        assert batch.shape[-1] == 3, (
            "expected image with 3 channels got {}".format(batch.shape[-1])
        )

        # If batch is actually just a single image with 3 dimensions give it
        # a batch dimension equal to 1
        if np.ndim(batch) == 3:
            batch = np.expand_dims(batch, 0)

        # Dimensions
        batch_size, height, width, channels = batch.shape

        summary_list = []
        for idx in range(0, self.max_outputs):
            image = batch[idx]

            # We need the images in encoded format (bytes); to get that we
            # must save it to a byte stream...
            image_io = BytesIO()
            plt.imsave(image_io, image, format='png')

            # ...and get its contents after
            image_string_io = image_io.getvalue()
            image_io.close()

            # Create and append the summary to the list
            image_summary = tf.Summary.Image(
                height=height,
                width=width,
                colorspace=channels,
                encoded_image_string=image_string_io
            )
            image_tag = "{0}/{1}".format(tag, idx + 1)
            summary_list.append(
                tf.Summary.Value(tag=image_tag, image=image_summary)
            )

        return summary_list
