import tensorflow as tf

from model.model_fn import build_model
from model.input_fn import train_input_fn
from model.triplet_loss import batch_all_center_triplet_loss
from utils.train_utils import Params


if __name__ == '__main__':
    params = Params('model/parameters.json')
    with tf.variable_scope('model'):
        model = build_model(params)
    images, labels = train_input_fn(params)
    # latest_ckpt = tf.train.latest_checkpoint('D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints')

    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    config = tf.ConfigProto(gpu_options=gpu_options)
    with tf.Session(config=config) as sess:
        saver.restore(sess,
                      'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints/best/model.ckpt-3933')
        outputs = model(images)

        outputs_batch = outputs[:params.batch_size]
        outputs_ref = outputs[params.batch_size:]
        labels_batch = labels[:params.batch_size]
        _, distance = batch_all_center_triplet_loss(outputs, labels, params)
        predict_labels = tf.argmin(distance, axis=1)
        true_labels = labels[:params.batch_size]

        acc = tf.equal(tf.cast(predict_labels,tf.float32),
                       tf.cast(true_labels, tf.float32))
        # acc = tf.cast(acc, tf.float32)
        # acc = tf.reduce_sum(acc) / params.batch_size

        print(sess.run([predict_labels, true_labels]))
        # print(sess.run(true_labels))
        print(sess.run(acc))