from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy
import numpy as np
from keras.layers import Concatenate

_EPSILON = 1e-7
_epsilon = tf.convert_to_tensor(_EPSILON, dtype='float32')
class_weight = tf.convert_to_tensor([0.2, 1.8], dtype='float32')


def dice_coef_loss(y_true, y_pred, epsilon=1.0):
    # shape of y (batch,H*W,channel)
    # skip the batch and class axis for calculating Dice score
    axes = -2
    numerator = 2. * K.sum(y_pred * y_true, axes)
    numerator = tf.add(numerator, epsilon)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    channel_dice_loss = tf.divide(numerator, tf.add(denominator, epsilon))
    channel_dice_loss = tf.clip_by_value(channel_dice_loss, _epsilon,
                                         1 - _epsilon)
    channel_log_dice_loss = -tf.log(channel_dice_loss)
    channel_log_dice_loss = channel_log_dice_loss * class_weight
    dice_loss = K.mean(channel_log_dice_loss)  # average over classes and batch

    crossentropy_loss = categorical_crossentropy(y_true, y_pred)
    return 0.5 * dice_loss + 0.5 * crossentropy_loss


def lovasz_softmax_loss(batch_y):
    # shape of y (H*W,channel)
    y_true = batch_y[0]
    y_pred = batch_y[1]
    axes = -2
    number_of_class = K.int_shape(y_pred)[-1]
    errors = (y_true - y_pred)
    forground_mask = tf.cast(tf.greater(errors, 0.0), 'float32')
    positive_errors = tf.abs(errors)

    # (class)
    forground_counter = K.sum(forground_mask, axes)
    forground_counter = tf.cast(forground_counter, 'int32')

    losses = []
    for c in range(number_of_class):
        errors_sorted = tf.cond(
            K.any(forground_counter[c]), lambda: tf.nn.top_k(
                positive_errors[:, c], k=forground_counter[c])[0], lambda: tf.
            convert_to_tensor(np.array([0.0], dtype='float32')))

        class_losses = K.mean(errors_sorted)
        losses.append(class_losses)

    losses = tf.convert_to_tensor(losses, dtype='float32')
    losses = 1.0 - losses
    channel_loss = tf.clip_by_value(losses, _epsilon, 1 - _epsilon)
    channel_log_loss = -tf.log(channel_loss)
    img_loss = K.mean(channel_log_loss)
    return img_loss


def lovasz_softmax_loss_batch(y_true, y_pred):

    batch_y = K.concatenate([[y_true], [y_pred]], axis=1)
    losses = tf.map_fn(lovasz_softmax_loss, batch_y)
    lovaszsoftmax_loss = K.mean(losses)

    dicecoef_loss = dice_coef_loss(y_true, y_pred)
    return dicecoef_loss * 0.66 + lovaszsoftmax_loss * 0.33


def hard_example_mining(batch_y):

    # shape of y (H*W,channel)
    y_true = batch_y[0]
    y_pred = batch_y[1]
    axes = -2
    number_of_class = K.int_shape(y_pred)[-1]
    errors = (y_true - y_pred)
    forground_mask = tf.cast(tf.greater(errors, 0.0), 'float32')
    positive_errors = tf.abs(errors)

    # (class)
    forground_counter = K.sum(forground_mask, axes)
    forground_counter = tf.cast(forground_counter, 'int32')

    losses = []
    for c in range(number_of_class):
        errors_sorted = tf.cond(
            K.any(forground_counter[c]), lambda: tf.nn.top_k(
                positive_errors[:, c], k=forground_counter[c])[0], lambda: tf.
            convert_to_tensor(np.array([0.0], dtype='float32')))

        class_losses = K.mean(errors_sorted)
        losses.append(class_losses)

    channel_hard_losses = tf.convert_to_tensor(losses, dtype='float32')

    epsilon = 1.0
    numerator = 2. * K.sum(y_pred * y_true, axes)
    numerator = tf.add(numerator, epsilon)
    denominator = K.sum(K.square(y_pred) + K.square(y_true), axes)

    channel_dice_coefficient = tf.divide(numerator,
                                         tf.add(denominator, epsilon))

    hard_dice_loss = channel_hard_losses * (1.0 - channel_dice_coefficient)
    hard_dice_loss = class_weight * hard_dice_loss

    losses = 1.0 - hard_dice_loss
    channel_loss = tf.clip_by_value(losses, _epsilon, 1 - _epsilon)
    channel_log_loss = -tf.log(channel_loss)
    img_loss = K.mean(channel_log_loss)
    return img_loss


def hard_example_mining_batch(y_true, y_pred):

    batch_y = K.concatenate([[y_true], [y_pred]], axis=1)
    losses = tf.map_fn(hard_example_mining, batch_y)
    batch_loss = K.mean(losses)

    #crossentropy_loss = categorical_crossentropy(y_true, y_pred)
    #return crossentropy_loss * batch_loss
    return batch_loss
