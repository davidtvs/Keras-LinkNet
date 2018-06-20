import tensorflow as tf
import numpy as np


def weighted_categorical_crossentropy(class_weights):
    if isinstance(class_weights, dict):
        np_weights = np.zeros((len(class_weights)))
        for key in sorted(class_weights.keys()):
            np_weights[key] = class_weights[key]
    elif isinstance(class_weights, list) or isinstance(class_weights, tuple):
        np_weights = np.ndarray(class_weights)
    elif isinstance(class_weights, np.ndarray):
        np_weights = class_weights
    else:
        raise ValueError(
            "'class_weights' must of type: dict, list, tuple, "
            "or np.ndarray. Got: {}".format(type(class_weights))
        )

    tf_weights = tf.convert_to_tensor(np_weights, np.float32)
    print(tf_weights)

    def run(y_true, y_pred):
        # scale preds so that the class probas of each sample sum to 1
        y_pred /= tf.reduce_sum(y_pred, -1, True)
        # manual computation of crossentropy
        _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
        output = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(tf.multiply(y_true * tf.log(output), tf_weights), -1)

    return run
