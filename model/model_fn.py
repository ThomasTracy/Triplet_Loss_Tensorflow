''' Define the model '''

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, \
    MaxPool2D, Dense, Flatten, ReLU

from model.triplet_loss import batch_all_triplet_loss, batch_hard_triplet_loss, batch_all_center_triplet_loss
from utils.train_utils import Params, ProjectorSaverHook
from tensorflow.contrib.tensorboard.plugins import projector

def build_model(params):

    """

    :return: features (batch_size, feature_size)
    """
    with tf.variable_scope('feature_extractor'):
        feature_extractor = tf.keras.Sequential([
            Conv2D(input_shape=(params.image_size, params.image_size, params.image_channel),
                   filters=64, kernel_size=5,  # Converted to grayscale
                   padding='same', data_format="channels_last"),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=128, kernel_size=5, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=256, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),
            MaxPool2D((2, 2)),

            Conv2D(filters=512, kernel_size=3, padding='same'),
            BatchNormalization(),
            ReLU(),

            Flatten(),
            # Dense(512, name='fully_connect_1', activation='relu'),
            Dense(params.embedding_size, name='fully_connect')
        ])
        return feature_extractor


def model_fn(features, labels, mode, params):
    images = features

    is_training = (mode==tf.estimator.ModeKeys.TRAIN)



    assert images.shape[1:] == [params.image_size, params.image_size, 3], 'Wrong image size {}'.format(images.shape())

    # -----------------------------------------
    #         Define model's structure
    # -----------------------------------------

    with tf.variable_scope('model'):
        feature_extractor = build_model(params)

        # Finetune or not
        if params.finetune:
            if tf.gfile.IsDirectory(params.finetune_path):
                ckpt_path = tf.train.latest_checkpoint(params.finetune_path)
                variables_to_restore = tf.contrib.slim.get_variables_to_restore()
                tf.train.init_from_checkpoint(ckpt_path,
                                              # {'/':'model/'})
                                              {v.name.split(':')[0]: v for v in variables_to_restore})
                print('restored checkpoints from {}'.format(ckpt_path))
                # print(variables_to_restore)
        features = feature_extractor(images)
    embedding_var = tf.Variable(tf.zeros([params.batch_size+params.references_num, params.embedding_size]),
                                name='projector_embeddings')
    embedding_assign = tf.assign(embedding_var, features)
    # print("shape_embedding_var", embedding_var.shape)
    # print(images.shape)
    # print(features.shape)

    # 2-Norm, Frobenius norm of matrix
    # txt_features = tf.Print(tf.as_string(features),
    #                         [tf.as_string(features)],
    #                         message='features',
    #                         name='features')
    features_mean_norm = tf.reduce_mean(tf.norm(features, axis=1))

    labels = tf.cast(labels, tf.int32)
    labels_batch = labels[:params.batch_size]

    # --------------------------------------------
    #           Define loss and predictions
    # --------------------------------------------
    if params.triplet_strategy == 'batch_all':
        loss, distance = batch_all_center_triplet_loss(params, features, labels)
        # loss, frac = batch_all_triplet_loss(features, labels, params.margin, params.squared)
    elif params.triplet_strategy == 'batch_hard':
        loss = batch_hard_triplet_loss(features, labels, params.margin, params.squared)
    else:
        raise ValueError("Don't exist such kind of tiplet loss: {}".format(params.triplet_strategy))
    predict_label = tf.argmin(distance, axis=1)

    accuracy = tf.equal(tf.squeeze(tf.cast(predict_label, tf.float32)),
                        tf.squeeze(tf.cast(labels_batch, tf.float32)))
    accuracy = tf.reduce_sum(tf.cast(accuracy, tf.float32)) / params.batch_size

    logging_hook = tf.train.LoggingTensorHook(
        {"loss": loss, "accuracy": accuracy},
        every_n_iter=10
    )
    # summary_hook = tf.train.SummarySaverHook(save_steps=params.save_summary_steps,
    #                                          output_dir='D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\logs',
    #                                          scaffold=tf.train.Scaffold(summary_op=tf.summary.merge_all()))
    projector_hook = ProjectorSaverHook(images, labels, embedding_assign)
    # -----------------------------------------
    #         Define metrics and summary
    # -----------------------------------------
    with tf.variable_scope('metrics'):
        eval_metrics_ops = {
            'feature_mean_norm': tf.metrics.mean(features_mean_norm),
            'accuracy': tf.metrics.accuracy(labels=labels_batch,
                                            predictions=predict_label, name='acc_op')
        }

        # if params.triplet_strategy =='batch_all':
        #     eval_metrics_ops['fraction_positive_triplets'] = tf.metrics.mean(frac)

    txt_features = tf.as_string(features[0:5])
    embedding_text = tf.as_string(embedding_var[0:5])
    tf.summary.text('features', txt_features)
    tf.summary.text('embedding', embedding_text)
    tf.summary.histogram('embeddings', features)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('features_mean_norm', features_mean_norm)
    # tf.summary.scalar('accuracy', accuracy[1])
    # if params.triplet_strategy == 'batch_all':
    #     tf.summary.scalar('fraction_positive_triplets', frac)
    tf.summary.image('train_image', images, max_outputs=1)

    # Summary for projector
    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = params.meta_file
    embedding.sprite.image_path = params.sprite_path
    embedding.sprite.single_image_dim.extend([64, 64])
    projector.visualize_embeddings(tf.summary.FileWriter(params.model_dir), config)

    # -----------------------------------------
    # Define predict, eval, train EstimatorSpec
    # -----------------------------------------
    # 预测值为feature，至于该feature属于哪一类别需要后期处理
    if mode == tf.estimator.ModeKeys.PREDICT:
        # predictions = {'features': features}
        predictions = {
            'embeddings': features,
            'predict_labels': predict_label
        }
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics_ops)

    # optimizer = tf.train.AdamOptimizer(params.learning_rate)
    optimizer = tf.train.RMSPropOptimizer(params.learning_rate)
    global_step = tf.train.get_or_create_global_step()
    print("Now global_step is:", global_step)
    if params.use_batch_norm:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_ops = optimizer.minimize(loss, global_step=global_step)
    else:
        train_ops = optimizer.minimize(loss, global_step=global_step)

    return tf.estimator.EstimatorSpec(mode, loss=loss,
                                      train_op=train_ops,
                                      training_hooks=[logging_hook, projector_hook])


if __name__ == '__main__':
    params = Params('../model/parameters.json')
    input = tf.ones(shape=[1,64,64,3])
    feature_extractor = build_model(params)
    feature_extractor.summary()
    out = feature_extractor(input)
    init = tf.global_variables_initializer()
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        sess.run(init)
        print(sess.run(out))