import tensorflow as tf
import cv2
import numpy as np

from matplotlib import pyplot as plt

from model.model_fn import build_model
from model.input_fn import train_input_fn, test_input_fn
from model.triplet_loss import batch_all_center_triplet_loss
from utils.train_utils import Params


def test_ckpt():
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
        _, distance = batch_all_center_triplet_loss(params, outputs, labels)
        predict_labels = tf.argmin(distance, axis=1)
        true_labels = labels[:params.batch_size]

        acc = tf.equal(tf.cast(predict_labels, tf.float32),
                       tf.cast(true_labels, tf.float32))
        # acc = tf.cast(acc, tf.float32)
        # acc = tf.reduce_sum(acc) / params.batch_size

        print(sess.run([predict_labels, true_labels]))
        # print(sess.run(true_labels))
        print(sess.run(acc))


def test_pb(input_image,
            frozen_graph_path='D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints\\frozen_graph_full\\frozen_inference_graph.pb'):
    params = Params('model/parameters.json')
    frozen_graph_path = frozen_graph_path
    model_graph = tf.Graph()

    # image1 = cv2.imread('D:\\Data\\TrafficSigns\\test\\36_1.jpg')
    # image1 = cv2.resize(image1, (params.image_size, params.image_size))
    # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    # image1 = image1/ 255.0
    #
    # image = [image1, image2, image3]

    with model_graph.as_default():
        # input_image_ = tf.identity(input_image)
        od_graph_def = tf.GraphDef()

        image_input= test_input_fn(input_image, params)
        with tf.gfile.GFile(frozen_graph_path, 'rb') as f:
            serialized_graph = f.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with model_graph.as_default():
        with tf.Session(graph=model_graph) as sess:
            inputs = model_graph.get_tensor_by_name('image_input:0')
            outputs = model_graph.get_tensor_by_name('classes:0')
            image_input = sess.run(image_input)

            result = sess.run(outputs, feed_dict={inputs:image_input})
            # distance = batch_all_center_triplet_loss(params, outputs)
            # predict_labels = tf.argmin(distance, axis=1)
            # result, outputs = sess.run([predict_labels,outputs], feed_dict={inputs:image_input})
            # np.savetxt('D:/Data/test_result/outputs1.txt', outputs, fmt='%f', delimiter=',')
            # result = sess.run(outputs, feed_dict={inputs:image})
            print(result)
            # print(labels)
    return result





if __name__ == '__main__':
    img = cv2.imread('D:/Data/TrafficSigns/test/1.png')
    img = cv2.resize(img, (64, 64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img/255.0
    result = test_pb(img)
    print(result)

    # x0 = 135.67654418945312
    # x1 = 173.3216552734375
    # y0 = 3208.3896484375
    # y1 = 3218.89697265625
    #
    # print(img.shape)
    # img_croped = img[int(y0):int(y1), int(x0):int(x1)]
    # print(img_croped.shape)