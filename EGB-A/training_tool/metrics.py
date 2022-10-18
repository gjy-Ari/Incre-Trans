from keras import backend as K
import tensorflow as tf


#https://github.com/Golbstein/KerasExtras/blob/master/keras_functions.py
def m_iou(y_true, y_pred):
    epsilon = 10
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(0, nb_classes
                   ):  # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        # ious = K.sum(inter, axis=1)/K.sum(union, axis=1)
        ious = tf.add(K.sum(inter, axis=1), epsilon) / tf.add(
            K.sum(union, axis=1), epsilon)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(
            legal_batches))))  # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def m_iou_0(y_true, y_pred):
    epsilon = 10
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(
            0, nb_classes -
            1):  # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        ious = tf.add(K.sum(inter, axis=1), epsilon) / tf.add(
            K.sum(union, axis=1), epsilon)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(
            legal_batches))))  # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    return K.mean(iou)


def m_iou_1(y_true, y_pred):
    epsilon = 10
    nb_classes = K.int_shape(y_pred)[-1]
    iou = []
    true_pixels = K.argmax(y_true, axis=-1)
    pred_pixels = K.argmax(y_pred, axis=-1)
    void_labels = K.equal(K.sum(y_true, axis=-1), 0)
    for i in range(nb_classes - 1, nb_classes
                   ):  # exclude first label (background) and last label (void)
        true_labels = K.equal(true_pixels, i) & ~void_labels
        pred_labels = K.equal(pred_pixels, i) & ~void_labels
        inter = tf.to_int32(true_labels & pred_labels)
        union = tf.to_int32(true_labels | pred_labels)
        legal_batches = K.sum(tf.to_int32(true_labels), axis=1) > 0
        ious = tf.add(K.sum(inter, axis=1), epsilon) / tf.add(
            K.sum(union, axis=1), epsilon)
        iou.append(K.mean(tf.gather(ious, indices=tf.where(
            legal_batches))))  # returns average IoU of the same objects
    iou = tf.stack(iou)
    legal_labels = ~tf.debugging.is_nan(iou)
    iou = tf.gather(iou, indices=tf.where(legal_labels))
    if iou is None:
        return 1.0
    else:
        return K.mean(iou)