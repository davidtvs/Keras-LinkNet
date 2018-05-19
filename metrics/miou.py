import tensorflow as tf
import numpy as np


class MeanIoU(object):
    """Mean intersection over union (mIoU) metric.

    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:

        IoU = true_positive / (true_positive + false_positive + false_negative).

    The mean IoU is the mean of IoU between all classes.

    Keyword arguments:
        num_classes (int): number of classes in the classification problem.
        ignore (int or tuple): a single integer or tuple of intergers in the
            range [0, `num_classes` - 1] identifying the classes to ignore when
            computing the mean IoU. Default: None, no class is ignored.

    """

    def __init__(self, num_classes, ignore=None):
        super().__init__()

        self.num_classes = num_classes

        # Handle the three types of input for ignore: None, int, and iterable
        if ignore is None:
            self.ignore = None
        elif isinstance(ignore, int):
            # We are going to make a mask of classes to ignore. Therefore, we
            # make an array with 'num_classes' elements and set elements
            # to True if we want to ignore the class and False otherwise
            self.ignore = np.zeros((num_classes), dtype=bool)
            self.ignore[ignore] = True
        else:
            # Same as above, but now we have an iterable inestead of an int
            self.ignore = np.zeros((num_classes), dtype=bool)
            for i in ignore:
                if 0 <= i < num_classes:
                    self.ignore[i] = True
                else:
                    raise ValueError(
                        "ignore argument values not in range [0, {}]".
                        format(num_classes - 1)
                    )

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
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and
        # set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (
                true_positive + false_positive + false_negative
            )
        iou[np.isnan(iou)] = 0

        # Remove classes set to be ignored from IoU array
        if self.ignore is not None:
            iou = iou[~self.ignore]

        return np.mean(iou).astype(np.float32)
