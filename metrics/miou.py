import tensorflow as tf
import numpy as np


class MeanIoU(object):
    """Mean intersection over union (mIoU) metric.

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative)

    The mean IoU is the mean of IoU between all classes.

    Keyword arguments:
        num_classes (int): number of classes in the classification problem.
        ignore_index (int or iterable, optional): Index of the classes to
            ignore when computing the IoU. Can be an int, or any iterable of
            ints.

    """

    def __init__(self, num_classes, ignore_index=None):
        super().__init__()

        self.num_classes = num_classes

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index, )
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def mean_iou(self, y_true, y_pred):
        """The metric function to be passed to the model.

        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.

        Returns:
            The mean intersection over union as a tensor.

        """
        # Wraps _mean_iou function and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self._mean_iou, [y_true, y_pred], tf.float32)

    def _mean_iou(self, y_true, y_pred):
        """Computes the mean intesection over union using numpy.

        Args:
            y_true (tensor): True labels.
            y_pred (tensor): Predictions of the same shape as y_true.

        Returns:
            The mean intersection over union (np.float32).

        """
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        target = np.argmax(y_true, axis=-1).ravel()
        predicted = np.argmax(y_pred, axis=-1).ravel()

        # Trick for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int32), minlength=self.num_classes**2
        )
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        if self.ignore_index is not None:
            for index in self.ignore_index:
                conf[:, self.ignore_index] = 0

        np.set_printoptions(linewidth=300)
        print()
        print()
        print(conf)
        print()
        print()
        np.set_printoptions(edgeitems=3,infstr='inf', linewidth=75, nanstr='nan', precision=8, suppress=False, threshold=1000, formatter=None)

        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)  # yapf: disable

        return np.nanmean(iou).astype(np.float32)
