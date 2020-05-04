import numpy as np
import keras.backend as K
import tensorflow as tf
import configs as cfgs
from keras.losses import categorical_crossentropy

smooth = 1e-3

def get_metrics():
    metrics = []
    for mt in (cfgs.metrics):
        metrics.append(get_function(mt))
    return metrics

def get_loss():
    return get_function(cfgs.loss)

def get_function(name):
    if name == "dice_coef":
        return dice_coef
    elif name == "dice_coef_loss":
        return dice_coef_loss
    elif name == "dice_coef_loss_bce":
        return dice_coef_loss_bce
    elif name == "binary_crossentropy":
        return binary_crossentropy
    elif name == "double_head_loss":
        return double_head_loss
    elif name == "softmax_dice_loss":
        return softmax_dice_loss
    elif name == "tversky_loss":
        return tversky_loss
    elif name == "generalised_dice_loss":
        return generalised_dice_loss
    elif name == "tversky_nf":
        return tversky_nf
    elif name == "precision_th":
        return as_keras_metric(precision_th)
    elif name == "recall_th":
        return as_keras_metric(recall_th)
    elif name == "f1_th":
        return as_keras_metric(f1_th)
    elif name == "hard_dice_coef_th":
        return hard_dice_coef_th
    elif name == "precision":
        return as_keras_metric(precision)
    elif name == "recall":
        return as_keras_metric(recall)
    elif name == "f1":
        return as_keras_metric(f1)
    elif name == "hard_dice_coef":
        return hard_dice_coef
    elif name == "accuracy":
        return accuracy
    else:
        raise Exception("Failed get metric/loss name: {:}".format(name))

# Loss function
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return K.mean((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

def dice_coef_loss_bce(y_true, y_pred, dice=0.5, bce=0.5):
    return binary_crossentropy(y_true, y_pred) * bce + dice_coef_loss(y_true, y_pred) * dice

def binary_crossentropy(y, p):
    return K.mean(K.binary_crossentropy(y, p))

def double_head_loss(y_true, y_pred):
    mask_loss = dice_coef_loss_bce(y_true[..., 0], y_pred[..., 0])
    contour_loss = dice_coef_loss_bce(y_true[..., 1], y_pred[..., 1])
    return mask_loss + contour_loss

def softmax_dice_loss(y_true, y_pred):
    return categorical_crossentropy(y_true, y_pred) * 0.6 + dice_coef_loss(y_true[..., 0], y_pred[..., 0]) * 0.2 + dice_coef_loss(y_true[..., 1], y_pred[..., 1]) * 0.2

def tversky_loss(y_true, y_pred):
    alpha = 0.5
    beta  = 0.5
    
    ones = K.ones(K.shape(y_true))
    p0 = y_pred      # proba that voxels are class i
    p1 = ones-y_pred # proba that voxels are not class i
    g0 = y_true
    g1 = ones-y_true
    
    num = K.sum(p0*g0, (0,1,2))
    den = num + alpha*K.sum(p0*g1,(0,1,2)) + beta*K.sum(p1*g0,(0,1,2))
    
    T = K.sum(num/den) # when summing over classes, T has dynamic range [0 Ncl]
    
    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl-T

def labels_to_one_hot(ground_truth, num_classes=1):
    """
    Converts ground truth labels to one-hot, sparse tensors.
    Used extensively in segmentation losses.
    :param ground_truth: ground truth categorical labels (rank `N`)
    :param num_classes: A scalar defining the depth of the one hot dimension
        (see `depth` of `tf.one_hot`)
    :return: one-hot sparse tf tensor
        (rank `N+1`; new axis appended at the end)
    """
    num_classes = cfgs.n_classes

    # read input/output shapes
    if isinstance(num_classes, tf.Tensor):
        num_classes_tf = tf.to_int32(num_classes)
    else:
        num_classes_tf = tf.constant(num_classes, tf.int32)
    input_shape = tf.shape(ground_truth)
    output_shape = tf.concat(
        [input_shape, tf.reshape(num_classes_tf, (1,))], 0)

    if num_classes == 1:
        # need a sparse representation?
        return tf.reshape(ground_truth, output_shape)

    # squeeze the spatial shape
    ground_truth = tf.reshape(ground_truth, (-1,))
    # shape of squeezed output
    dense_shape = tf.stack([tf.shape(ground_truth)[0], num_classes_tf], 0)

    # create a rank-2 sparse tensor
    ground_truth = tf.to_int64(ground_truth)
    ids = tf.range(tf.to_int64(dense_shape[0]), dtype=tf.int64)
    ids = tf.stack([ids, ground_truth], axis=1)
    one_hot = tf.SparseTensor(
        indices=ids,
        values=tf.ones_like(ground_truth, dtype=tf.float32),
        dense_shape=tf.to_int64(dense_shape))

    # resume the spatial dims
    one_hot = tf.sparse_reshape(one_hot, output_shape)
    return one_hot

def generalised_dice_loss(y_true,
                          y_pred,
                          weight_map=None,
                          type_weight='Simple'):
    """
    Function to calculate the Generalised Dice Loss defined in
        Sudre, C. et. al. (2017) Generalised Dice overlap as a deep learning
        loss function for highly unbalanced segmentations. DLMIA 2017
    :param prediction: the logits
    :param ground_truth: the segmentation ground truth
    :param weight_map:
    :param type_weight: type of weighting allowed between labels (choice
        between Square (square of inverse of volume),
        Simple (inverse of volume) and Uniform (no weighting))
    :return: the loss
    """

    ground_truth = y_true
    prediction = y_pred

    prediction = tf.cast(prediction, tf.float32)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        # weight_map_nclasses = tf.reshape(
        #     tf.tile(weight_map, [num_classes]), prediction.get_shape())
        weight_map_nclasses = tf.tile(
            tf.expand_dims(tf.reshape(weight_map, [-1]), 1), [1, num_classes])
        ref_vol = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot, reduction_axes=[0])

        intersect = tf.sparse_reduce_sum(
            weight_map_nclasses * one_hot * prediction, reduction_axes=[0])
        seg_vol = tf.reduce_sum(
            tf.multiply(weight_map_nclasses, prediction), 0)
    else:
        ref_vol = tf.sparse_reduce_sum(one_hot, reduction_axes=[0])
        intersect = tf.sparse_reduce_sum(one_hot * prediction,
                                         reduction_axes=[0])
        seg_vol = tf.reduce_sum(prediction, 0)
    if type_weight == 'Square':
        weights = tf.reciprocal(tf.square(ref_vol))
    elif type_weight == 'Simple':
        weights = tf.reciprocal(ref_vol)
    elif type_weight == 'Uniform':
        weights = tf.ones_like(ref_vol)
    else:
        raise ValueError("The variable type_weight \"{}\""
                         "is not defined.".format(type_weight))
    new_weights = tf.where(tf.is_inf(weights), tf.zeros_like(weights), weights)
    weights = tf.where(tf.is_inf(weights), tf.ones_like(weights) *
                       tf.reduce_max(new_weights), weights)
    generalised_dice_numerator = \
        2 * tf.reduce_sum(tf.multiply(weights, intersect))
    # generalised_dice_denominator = \
    #     tf.reduce_sum(tf.multiply(weights, seg_vol + ref_vol)) + 1e-6
    generalised_dice_denominator = tf.reduce_sum(
        tf.multiply(weights, tf.maximum(seg_vol + ref_vol, 1)))
    generalised_dice_score = \
        generalised_dice_numerator / generalised_dice_denominator
    generalised_dice_score = tf.where(tf.is_nan(generalised_dice_score), 1.0,
                                      generalised_dice_score)
    return 1 - generalised_dice_score

def tversky_nf(y_true, y_pred, weight_map=None, alpha=0.5, beta=0.5):
    """
    Function to calculate the Tversky loss for imbalanced data
        Sadegh et al. (2017)
        Tversky loss function for image segmentation
        using 3D fully convolutional deep networks
    :param prediction: the logits
    :param ground_truth: the segmentation ground_truth
    :param alpha: weight of false positives
    :param beta: weight of false negatives
    :param weight_map:
    :return: the loss
    """

    ground_truth = y_true
    prediction = y_pred

    prediction = tf.to_float(prediction)
    if len(ground_truth.shape) == len(prediction.shape):
        ground_truth = ground_truth[..., -1]
    one_hot = labels_to_one_hot(ground_truth, tf.shape(prediction)[-1])
    one_hot = tf.sparse_tensor_to_dense(one_hot)

    p0 = prediction
    p1 = 1 - prediction
    g0 = one_hot
    g1 = 1 - one_hot

    if weight_map is not None:
        num_classes = prediction.shape[1].value
        weight_map_flattened = tf.reshape(weight_map, [-1])
        weight_map_expanded = tf.expand_dims(weight_map_flattened, 1)
        weight_map_nclasses = tf.tile(weight_map_expanded, [1, num_classes])
    else:
        weight_map_nclasses = 1

    tp = tf.reduce_sum(weight_map_nclasses * p0 * g0)
    fp = alpha * tf.reduce_sum(weight_map_nclasses * p0 * g1)
    fn = beta * tf.reduce_sum(weight_map_nclasses * p1 * g0)

    EPSILON = 0.00001
    numerator = tp
    denominator = tp + fp + fn + EPSILON
    score = numerator / denominator
    return 1.0 - tf.reduce_mean(score)

# Metrics
def as_keras_metric(method):
    import functools
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

def hard_dice_coef(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.round(y_pred[..., 1]))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def hard_dice_coef_th(y_true, y_pred):
    y_true_f = K.flatten(K.round(y_true[..., 1]))
    y_pred_f = K.flatten(K.cast(y_pred[..., 1] > cfgs.threshold, 'float32'))
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def precision_th(y_true, y_pred):
    y_true = tf.cast((y_true[..., 1] > cfgs.threshold), dtype=tf.float32)
    y_pred = tf.cast((y_pred[..., 1] > cfgs.threshold), dtype=tf.float32)
    return tf.metrics.precision(y_true, y_pred)

def recall_th(y_true, y_pred):
    y_true = tf.cast((y_true[..., 1] > cfgs.threshold), dtype=tf.float32)
    y_pred = tf.cast((y_pred[..., 1] > cfgs.threshold), dtype=tf.float32)
    return tf.metrics.recall(y_true, y_pred)

def f1_th(y_true, y_pred):
    y_true = tf.cast((y_true[..., 1] > cfgs.threshold), dtype=tf.float32)
    y_pred = tf.cast((y_pred[..., 1] > cfgs.threshold), dtype=tf.float32)
    return tf.contrib.metrics.f1_score(y_true, y_pred)

def precision(y_true, y_pred):
    return tf.metrics.precision(y_true[..., 1], y_pred[..., 1])

def recall(y_true, y_pred):
    return tf.metrics.recall(y_true[..., 1], y_pred[..., 1])

def f1(y_true, y_pred):
    return tf.contrib.metrics.f1_score(y_true[..., 1], y_pred[..., 1])

def accuracy(y_true, y_pred):
    y_true = K.cast(y_true[..., 1], 'float32')
    y_pred = K.cast(y_pred[..., 1] > cfgs.threshold, 'float32')
    error = abs(K.sum(y_pred-y_true)/(K.sum(y_true)+smooth))
    error = tf.where(error > 1., 1., error)
    return 1.-error