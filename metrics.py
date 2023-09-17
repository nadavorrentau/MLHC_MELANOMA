import numpy as np
import tensorflow as tf
from tensorflow import reduce_sum, reduce_mean, logical_and, logical_or, cast
from tensorflow.python.keras import backend


# def jaccard_index(true_mask, predicted_mask):
#     """
#     Calculate the Jaccard Index (IoU) between two binary masks.
#     """
#
#     intersection = np.logical_and(true_mask, predicted_mask)
#     union = np.logical_or(true_mask, predicted_mask)
#
#     jac = np.sum(intersection) / np.sum(union)
#
#     return jac

def jaccard_index(true_mask, predicted_mask):
    """
    Calculate the Jaccard Index (IoU) between two binary masks.

    Args:
    - true_mask: True binary mask (Tensor).
    - predicted_mask: Predicted binary mask (Tensor).

    Returns:
    - jaccard: Jaccard Index between the two masks (Tensor).
    """
    true_mask = (true_mask >= 0.5)
    true_mask = tf.cast(true_mask, dtype=tf.bool)
    predicted_mask = (predicted_mask >= 0.5)
    predicted_mask = tf.cast(predicted_mask, dtype=tf.bool)
    intersection = tf.reduce_sum(tf.cast(tf.logical_and(true_mask, predicted_mask), dtype=tf.float32))
    union = tf.reduce_sum(tf.cast(tf.logical_or(true_mask, predicted_mask), dtype=tf.float32))

    jac = intersection / union

    return jac


def jaccard_loss(true_mask, predicted_mask, smooth=100):
    # jaccard loss was evaluated and not used
    intersection = tf.reduce_sum(true_mask * predicted_mask)
    union = tf.reduce_sum(true_mask + predicted_mask)
    jac = (intersection + smooth) / (union - intersection + smooth)
    jac_dist = 1 - jac
    return jac_dist


def dice_coefficient(y_true, y_pred, smooth=1):
    # Calculate the Dice coefficient between predicted masks and ground truth masks.

    intersection = reduce_sum(y_true * y_pred, axis=(1, 2, 3))
    union = reduce_sum(y_true + y_pred, axis=(1, 2, 3)) + smooth
    dice = reduce_mean((2.0 * intersection + smooth) / union, axis=0)
    return dice


def dice_loss(y_true, y_pred):
    # dice loss was evaluated and not used
    return 1 - dice_coefficient(y_true, y_pred)


def recall(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    possible_positives = backend.sum(backend.round(backend.clip(y_true, 0, 1)))
    rec = true_positives / (possible_positives + backend.epsilon())
    return rec


def precision(y_true, y_pred):
    true_positives = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)))
    predicted_positives = backend.sum(backend.round(backend.clip(y_pred, 0, 1)))
    prec = true_positives / (predicted_positives + backend.epsilon())
    return prec


def f1_metric(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + backend.epsilon()))
