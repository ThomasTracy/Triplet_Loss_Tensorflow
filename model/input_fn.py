import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
import numpy


IMG_SIZE = 64

def load_traffic_signs(txt_path):
    """
    Load path and labels of all traffic signs
    from a txt file
    """
    images = []
    labels = []
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    for l in lines:
        images.append(l.strip().split(',')[0])
        labels.append(int(l.strip().split(',')[1]))

    return images, labels

def load_img(img_path):
    img = tf.io.read_file(img_path)

    # Turn image in 64x64x3
    img = tf.image.decode_jpeg(img, channels=3)

    # Convert from (0,255) to (0,1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    return tf.image.resize(img, [IMG_SIZE, IMG_SIZE])

def normalize(img):

    # Standarize: (image - mean)/std
    # Standard mean and std from ImageNet
    img_mean = tf.constant([0.485, 0.456, 0.406])
    img_std = tf.constant([0.229, 0.224, 0.225])

    img_ = (img - img_mean)/img_std
    return img_

def preprocess(image, label):


    # labels = tf.cast(labels, tf.int32)
    # labels = [int(l) for l in labels]
    img = load_img(image)

    return img, label

def build_dataset(params):
    images, labels = load_traffic_signs('D:\Data\GTSRB\Final_Training\Images_jpg\\train.txt')

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.shuffle(params.train_size, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess)
    dataset = dataset.batch(params.batch_size, drop_remainder=True) # Make sure all batch are 64, divided evenly
    dataset = dataset.repeat(3)
    dataset = dataset.prefetch(1)

    return dataset

def show_dataset(dataset):
    """
    Use class Iterator to get each
    elements in Dataset
    """
    iterator = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        img, label = sess.run(iterator.get_next())

    for i in range(10):
        ax =plt.subplot(2,5,i+1)
        ax.imshow(img[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label[i])

    plt.show()


if __name__ == '__main__':
    dataset = build_dataset()

    show_dataset(dataset)


    # iterator = dataset.make_one_shot_iterator()
    # img, label = iterator.get_next()
    # writer = tf.summary.FileWriter('../logging/tensorboard')
    #
    # with tf.Session() as sess:
    #     summary = sess.run(tf.summary.image('img', img))
    #     writer.add_summary(summary)