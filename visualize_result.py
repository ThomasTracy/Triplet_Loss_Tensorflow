import tensorflow as tf
import argparse
import os
import pathlib
import shutil

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.contrib.tensorboard.plugins import projector

from model.input_fn import test_input_fn, input_fn, train_input_fn, train_input_fn_customized
from model.model_fn import model_fn
from test import test_pb

from utils.train_utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='D:\Pycharm\Projects\Triplet-Loss-Tensorflow\model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--sprite_filename', default='data/sprite_100.png',
                    help="Sprite image for the projector")


def visualize_results():
    sprite_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/sprite_100.png'
    meta_file = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/labels_100.tsv'
    log_dir = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging'
    params = Params('model/parameters.json')
    params.batch_size = 1000

    images, labels = train_input_fn(params)
    with tf.Session() as sess:
        images, labels = sess.run([images, labels])
    embeddings, _ = test_pb(images,
                           frozen_graph_path='D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints\\with_data_augumentation\\frozen_graph_full\\frozen_inference_graph.pb')
    # tf.Print(tf.as_string(embeddings),
    #                     [tf.as_string(embeddings)],
    #                     message='result',
    #                     name='result')

    create_sprite_image(images)
    create_meta_file(labels)
    y = tf.Variable(embeddings, name='projector_embeddings')
    summary_writer = tf.summary.FileWriter(log_dir)

    # 通过project.ProjectorConfig类来帮助生成日志文件
    config = projector.ProjectorConfig()
    # 增加一个需要可视化的bedding结果
    embedding = config.embeddings.add()
    # 指定这个embedding结果所对应的Tensorflow变量名称
    embedding.tensor_name = y.name

    # Specify where you find the metadata
    # 指定embedding结果所对应的原始数据信息。比如这里指定的就是每一张MNIST测试图片对应的真实类别。在单词向量中可以是单词ID对应的单词。
    # 这个文件是可选的，如果没有指定那么向量就没有标签。
    embedding.metadata_path = meta_file

    # Specify where you find the sprite (we will create this later)
    # 指定sprite 图像。这个也是可选的，如果没有提供sprite 图像，那么可视化的结果
    # 每一个点就是一个小困点，而不是具体的图片。
    embedding.sprite.image_path = sprite_path
    # 在提供sprite图像时，通过single_image_dim可以指定单张图片的大小。
    # 这将用于从sprite图像中截取正确的原始图片。
    embedding.sprite.single_image_dim.extend([64, 64])

    # Say that you want to visualise the embeddings
    # 将PROJECTOR所需要的内容写入日志文件。
    projector.visualize_embeddings(summary_writer, config)

    # 生成会话，初始化新声明的变量并将需要的日志信息写入文件。
    sess = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.save(sess, os.path.join(log_dir, "model.ckpt"), 1)

    # summary_writer.close()


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""

    sprite_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/sprite_100.png'
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # sprite图像可以理解成是小图片平成的大正方形矩阵，大正方形矩阵中的每一个元素就是原来的小图片。于是这个正方形的边长就是sqrt(n),其中n为小图片的数量。
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    # 使用全1来初始化最终的大图片。
    spriteimage = np.ones((img_h*n_plots, img_w*n_plots, 3))

    for i in range(n_plots):
        for j in range(n_plots):
            # 计算当前图片的编号
            this_filter = i*n_plots + j
            if this_filter < images.shape[0]:
                # 将当前小图片的内容复制到最终的sprite图像
                this_img = images[this_filter]
                spriteimage[i*img_h:(i + 1)*img_h,
                j*img_w:(j + 1)*img_w] = this_img

    plt.imsave(sprite_path, spriteimage)

    # return spriteimage


def create_meta_file(labels):

    labels_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/labels_100.tsv'

    with open(labels_path, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, label))


def write_sprite_images():

    log_dir = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging'
    sprite_dir = 'sprite_100.png'
    labels_file = "labels_100.tsv"

    params = Params('model/parameters.json')
    images, labels = input_fn(params)
    with tf.Session() as sess:
        images, labels = sess.run([images, labels])

    sprite_image = create_sprite_image(images)
    sprite_path = os.path.join(log_dir, sprite_dir)
    plt.imsave(sprite_path, sprite_image)
    plt.imshow(sprite_image)
    plt.show()

    labels_path = os.path.join(log_dir, labels_file)
    with open(labels_path, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, label))


if __name__ == '__main__':
    # write_sprite_images()

    visualize_results()
