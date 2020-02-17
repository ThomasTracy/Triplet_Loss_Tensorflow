import tensorflow as tf
import numpy as np


feature = tf.constant([[1,2,3],
                       [4,5,6],
                       [7,8,9],
                       [10,11,12]], dtype=tf.float32)

print(feature.shape)
dot_product = tf.matmul(feature, tf.transpose(feature))
square_norm = tf.diag_part(dot_product)

distance = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
#
# with tf.Session() as sess:
#     print(sess.run(tf.expand_dims(square_norm, 0)))
#     print('dot product:', sess.run(dot_product))
#     print('square norm:', sess.run(square_norm))
#     print('distance:', sess.run(distance))

def _pairwise_distance(features, squared=False):
    '''

    :param features: output feature from Net  (batch_size, feature_size)
    :param squared: when False then use normal euclidean distance, not squared
    :return: distance between features of all images   (batch_size, batch_size)
    '''

    dot_product = tf.matmul(features, tf.transpose(features))
    square_norm = tf.diag_part(dot_product)

    # In order to realize a^2 -2ab + b^2, the shape should be
    # square_norm expanded: (4, 1) - 2 * dot_product: (4, 4) + square_norm expanded: (1, 4)
    distance = tf.expand_dims(square_norm, 1) - 2.0 * dot_product + tf.expand_dims(square_norm, 0)
    distance = tf.maximum(distance, 0.0)

    if not squared:
        mask = tf.cast(tf.equal(distance, 0.0), tf.float32)
        distance = distance + mask * 1e-16
        distance = tf.sqrt(distance)
        distance = distance * (1.0 - mask)

    return distance

def _get_triplet_mask(labels):

    # (x, y, z) are different
    indice_equal = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    indice_not_equal = tf.logical_not(indice_equal)
    x_not_equal_y = tf.expand_dims(indice_not_equal, 2)
    x_not_equal_z = tf.expand_dims(indice_not_equal, 1)
    y_not_equal_z = tf.expand_dims(indice_not_equal, 0)
    mask_diff_indice = tf.logical_and(tf.logical_and(x_not_equal_y, x_not_equal_z), y_not_equal_z)

    # x = y , y != z  (positive, anchor, negative)
    same_labels = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    diff_labels = tf.logical_not(same_labels)
    x_y_same = tf.expand_dims(same_labels, 2)
    y_z_diff = tf.expand_dims(diff_labels, 0)
    mask_p_a_n = tf.logical_and(x_y_same, y_z_diff)

    # Combine
    mask_triplet = tf.logical_and(mask_diff_indice, mask_p_a_n)

    # Bool matrix
    return mask_triplet

def batch_all_triplet_loss(features, labels, margin, squared):

    pairwise_distance = _pairwise_distance(features, squared)

    # x - p - dim 0 | y - a - dim 1 | z - n - dim 2
    # distance positive-anchor is on plane XOY, expand dimension 2 (axis Z)
    # distance anchor-negative is on plane YOZ, expand dimension 0 (axis X)
    # loss(batch, batch, batch) = p_a(batch, batch, 1) - a_n(1, batch, batch) + margin
    p_a_distance = tf.expand_dims(pairwise_distance, 2)
    assert p_a_distance.shape[2] == 1, "Dimension 2 is not {}".format(p_a_distance.shape[2])
    a_n_distance = tf.expand_dims(pairwise_distance, 0)
    assert a_n_distance.shape[0] == 1, "Dimension 0 is not {}".format(a_n_distance.shape[0])

    triplet_loss = p_a_distance - a_n_distance + margin

    mask = _get_triplet_mask(labels)
    mask = tf.cast(mask, tf.float32)
    triplet_loss = tf.multiply(triplet_loss, mask)

    # Remove negative loss, i.e. easy triplets
    triplet_loss = tf.maximum(triplet_loss, 0.0)

    positive_triplets = tf.cast(tf.greater(triplet_loss, 1e-16), tf.float32)
    num_positive_triplets = tf.reduce_sum(positive_triplets)
    num_valid_triplets = tf.reduce_sum(mask)
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)

    # Mean triplet loss over positive triplets
    triplet_loss = triplet_loss / (num_positive_triplets + 1e-16)

    return triplet_loss, fraction_positive_triplets

def _get_p_a_mask(labels):


    mask_same_indices = tf.cast(tf.eye(tf.shape(labels)[0]), tf.bool)
    mask_diff_indices = tf.logical_not(mask_same_indices)

    mask_p_a = tf.equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))
    mask_p_a = tf.logical_and(mask_diff_indices, mask_p_a)

    # Returned is bool matrix
    return mask_p_a

def _get_a_n_mask(labels):

    mask_a_n = tf.not_equal(tf.expand_dims(labels, 0), tf.expand_dims(labels, 1))

    return mask_a_n

def batch_hard_triplet_loss(features, labels, squared):

    # Hardest positive distance -- largest positive distance
    distance = _pairwise_distance(features, squared)
    mask_p_a = _get_p_a_mask(labels)
    mask_p_a = tf.cast(mask_p_a, tf.float32)
    hard_positive_distance = tf.multiply(distance, mask_p_a)
    # shape (batch_size, 1)
    hard_positive_distance = tf.reduce_max(hard_positive_distance, 1, keepdims=True)
    tf.summary.scalar('hardest positive distance', tf.reduce_mean(hard_positive_distance))

    # Hardest negative distance -- smallest negative distance
    mask_a_n = _get_a_n_mask(labels)
    mask_a_n = tf.cast(mask_a_n, tf.float32)

    # Hear we cant use distance x mask
    # cause the 'not negative parts' will be 0
    # and the smallest distance will all be 0
    # so we directly add max data to all 'no negative parts'
    # then right 'negative distance' will be the smallest
    max_distance = tf.reduce_max(distance, 1, keepdims=True)
    hard_negative_distance = distance + max_distance * (1.0 - mask_a_n)
    # shape (batch_size, 1)
    hard_negative_distance = tf.reduce_min(hard_negative_distance, 1, keepdims=True)
    tf.summary.scalar('hardest negative distance', tf.reduce_mean(hard_negative_distance))

    hard_triplet_loss = tf.maximum(
                        hard_positive_distance - hard_negative_distance + margin, 0.0)
    hard_triplet_loss = tf.reduce_mean(hard_triplet_loss)

    return hard_triplet_loss




if __name__ == '__main__':
    # dis = _pairwise_distance(feature, False)
    labels = tf.constant([2,2,4,3])
    margin = 0.5
    mask = _get_p_a_mask(labels)
    hard_loss = batch_hard_triplet_loss(feature, labels, False)
    with tf.Session() as sess:
        print(sess.run(hard_loss))
