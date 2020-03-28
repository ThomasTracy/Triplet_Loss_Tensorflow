import os
import json
import time
import logging

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.contrib.tensorboard.plugins import projector


class Params():
    """
    Load hyperparameters from json file
    param = Params(json_path)
    param.batch_size --> 64
    """
    def __init__(self, json_path):
        self._update(json_path)

    def _update(self, json_path):

        # Load parameters from json file
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):

        # Save parameters into json file
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        # Allow visit parameters through dict way
        # param.dict["batch_size"]
        return self.__dict__


def create_sprite_image(images, path):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""

    # sprite_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/sprite_100.png'
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

    plt.imsave(path, spriteimage)
    print("\033[1;32m ------ Finish writting sprite image ------ \033[0m")


def create_meta_file(labels, path):

    # labels_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/labels_100.tsv'

    with open(path, 'w') as f:
        f.write("Index\tLabel\n")
        for index, label in enumerate(labels):
            f.write("%d\t%d\n" % (index, label))
    print("\033[1;32m ------ Finish writting meta file ------ \033[0m")


def write_projector(images, labels):
    sprite_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/sprite_100.png'
    metafile_path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging/labels_100.tsv'
    log_dir = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging'
    params = Params('model/parameters.json')

    # images, labels = train_input_fn(params)
    # with tf.Session() as sess:
    #     images, labels = sess.run([images, labels])
    create_sprite_image(images, sprite_path)
    create_meta_file(labels, metafile_path)

    final_result = test_pb(images,
                           frozen_graph_path='D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints\\frozen_graph\\frozen_inference_graph.pb')

    y = tf.Variable(final_result, name='embedding')
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
    embedding.metadata_path = metafile_path

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

    summary_writer.close()


class ProjectorSaverHook(tf.train.SessionRunHook):

    def __init__(self, images, labels, embedding_assign):
        super(ProjectorSaverHook, self).__init__()
        self.path = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints'
        self.images = images
        self.labels = labels
        self.embedding_assign = embedding_assign


    def begin(self):
        self.step = tf.train.get_or_create_global_step()


    # def after_create_session(self, session, coord):
        # self.saver(session, os.path.join(self.path, 'projector.ckpt'), self.step)
        # pass


    def before_run(self, run_context):
        return tf.train.SessionRunArgs([self.step, self.images, self.labels, self.embedding_assign])


    def after_run(self, run_context, run_values):
        if run_values.results[0] % 10 == 0:
            create_sprite_image(run_values.results[1],
                                os.path.join(self.path, 'sprite_img.png'))
            create_meta_file(run_values.results[2],
                             os.path.join(self.path, 'metafile.tsv'))

            # print(run_values.results[2])
            # print(run_values.results[3])
        # config = projector.ProjectorConfig()
        # embedding = config.embeddings.add()
        # embedding.tensor_name = 'projector_embeddings'
        # embedding.metadata_path = "D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints/metafile.tsv"
        # embedding.sprite.image_path = "D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints/sprite_img.png"
        # embedding.sprite.single_image_dim.extend([64, 64])
        # projector.visualize_embeddings(tf.summary.FileWriter('D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints'),
        #                                config)


    def end(self, session):
        pass


def set_logger(log_path):

    log_name = time.strftime('%Y%m%d_%H-%M-%S', time.localtime(time.time()))
    log_name = 'logging_' + log_name
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:

        # Logging to file
        file_handler = logging.FileHandler(os.path.join(log_path, log_name))
        file_handler. setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


if __name__ == '__main__':
    json_file = '../model/parameters.json'
    params = Params(json_file)
    print(params.learning_rate)

    set_logger('../logging')
    logging.info('now haha')
    test()