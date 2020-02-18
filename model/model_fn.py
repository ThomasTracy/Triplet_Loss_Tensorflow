''' Define the model '''

import tensorflow as tf

from model.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss


def build_model():

    """

    :return: features (batch_size, feature_size)
    """


def model_fn(images, labels, mode, params):

    is_training = (mode==tf.estimator.ModeKeys.TRAIN)
    assert images.shape[1:] == [params.image_size, params.image_size, 3], 'Wrong image size {}'.format(images.shape())

    # -----------------------------------
    # Define model's structure
    # -----------------------------------
    with tf.variable_scope('model'):
        features = build_model(images, is_training, params)
    # 2-Norm, Frobenius norm of matrix
    features_mean_norm = tf.reduce_mean(tf.norm(features, axis=1))
    tf.summary.scalar('features_mean_norm', features_mean_norm)

    # 预测值为feature，至于该feature属于哪一类别需要后期处理
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'features', features}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    labels = tf.cast(labels, tf.int32)

    # -----------------------------------
    # Define triplet loss
    # -----------------------------------
    if params.triplet_strategy == 'batch_all':
        loss, frac = batch_all_triplet_loss(features, labels, params.margin, params.squared)
    elif params.triplet_strategy == 'batch_hard':
        loss = batch_hard_triplet_loss(features, labels, params.margin, params.squared)
    else:
        raise ValueError("Don't exist such kind of tiplet loss: {}".format(params.triplet_strategy))

    # -----------------------------------
    # Define metrics and summary
    # -----------------------------------
    with tf.variable_scope('metrics'):
        eval_metrics_ops = {'feature_mean_norm': tf.metrics.mean(features_mean_norm)}

        if params.triplet_strategy = 'batch_all':
            eval_metrics_ops['fraction_positive_triplets'] = tf.metrics.mean(frac)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metrics_ops=eval_metrics_ops)

    tf.summary.scalar('loss', loss)
    if params.triplet_strategy = 'batch_all':
       tf.summary.scalar('fraction_positive_triplets', frac)
    tf.summary.image('train_image', images, max_outputs=1)

    optimizer = tf.train.AdamOptimizer(params.learning_rate)
    global_step = tf.train.get_global_step()
    if params.use_batch_norm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops = optimizer.minimize(loss, global_step=global_step)
    else:
        train_ops = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_ops)


if __name__ == '__main__':
    a = tf.constant([[1.,2.,3.], [4.,5.,6.]])
    b = tf.norm(a, axis=1)
    with tf.Session() as sess:
        print(sess.run(b))