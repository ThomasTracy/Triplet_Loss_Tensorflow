import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os

from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.examples.tutorials.mnist import input_data


LOG_DIR = 'D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging_temp'
NAME_TO_VISUALISE_VARIABLE = "mnistembedding"
TO_EMBED_COUNT = 500


def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots))

    for i in range(n_plots):
        for j in range(n_plots):
            this_filter = i * n_plots + j
            if this_filter < images.shape[0]:
                this_img = images[this_filter]
                spriteimage[i * img_h:(i + 1) * img_h,
                j * img_w:(j + 1) * img_w] = this_img

    return spriteimage


def vector_to_matrix_mnist(mnist_digits):
    """Reshapes normal mnist digit (batch,28*28) to matrix (batch,28,28)"""
    return np.reshape(mnist_digits, (-1, 28, 28))


def invert_grayscale(mnist_digits):
    """ Makes black white, and white black """
    return 1 - mnist_digits

path_for_mnist_sprites = os.path.join(LOG_DIR, 'mnistdigits.png')
path_for_mnist_metadata = os.path.join(LOG_DIR, 'metadata.tsv')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=False)
batch_xs, batch_ys = mnist.train.next_batch(TO_EMBED_COUNT)

embedding_var = tf.Variable(batch_xs, name=NAME_TO_VISUALISE_VARIABLE)

tf.summary.histogram("Embedding", embedding_var)
tf.summary.tensor_summary("Embeddings", embedding_var)
summary_writer = tf.summary.FileWriter(LOG_DIR)
# merged_summ = tf.summary.merge_all()

to_visualise = batch_xs
to_visualise = vector_to_matrix_mnist(to_visualise)
to_visualise = invert_grayscale(to_visualise)

sprite_image = create_sprite_image(to_visualise)

plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')
plt.imshow(sprite_image, cmap='gray')

with open(path_for_mnist_metadata, 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(batch_ys):
        f.write("%d\t%d\n" % (index, label))

config = projector.ProjectorConfig()
embedding = config.embeddings.add()
embedding.tensor_name = embedding_var.name

# Specify where you find the metadata
embedding.metadata_path = os.path.join(LOG_DIR, 'metadata.tsv')  # 'metadata.tsv'

# Specify where you find the sprite (we will create this later)
embedding.sprite.image_path = os.path.join(LOG_DIR, 'mnistdigits.png')  # 'mnistdigits.png'
embedding.sprite.single_image_dim.extend([28, 28])

# Say that you want to visualise the embeddings
projector.visualize_embeddings(summary_writer, config)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), 1)
#
# summary_writer.close()
#
# import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.contrib.tensorboard.plugins import projector #导入一个projector的包
#
# #载入数据集
# mnist=input_data.read_data_sets('mnist_data',one_hot=True)#one_hot把像素点都转变成0或1的形式
# #运行次数
# max_steps=2
# #图片数量
# image_num=100
# #文件路径
# DIR='D:/Pycharm/Projects/Triplet-Loss-Tensorflow/logging_temp/'
#
# #定义会话
# sess=tf.Session()
#
# #载入图片
# embedding=tf.Variable(tf.stack(mnist.test.images[:image_num]),trainable=False,name='embedding')
# #打包的是手写数字test里面的图片0-image_num，然后存在embedding
# '''
# stack函数：把各个数组打包
# x is [1，4]
# y is [2，5]
# z is [3，6]
# stack([x,y,z]) ==> [[1,4],[2,5],[3,6]]  axis默认为0，即横向打包
# stack([x,y,z],axis=1) ==>[[1,2,3],[4,5,6]]  axis为1时，即纵向打包
# '''
#
# #参数概要
# def variable_summaries(var):
#     with tf.name_scope('summaries'):
#         mean=tf.reduce_mean(var)#计算平均值
#         tf.summary.scalar('mean',mean)#记录平均值，将其命名为mean。summary.scalar用来显示标量信息
#         with tf.name_scope('stddev'):
#             stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
#         tf.summary.scalar('stddev',stddev)# 标准差
#         tf.summary.scalar('max',tf.reduce_max(var))#最大值
#         tf.summary.scalar('min',tf.reduce_min(var))#最小值
#         tf.summary.histogram('histogram',var)#直方图
#
# #命名空间
# with tf.name_scope('input'):  #命名随意，比如input,下面的x和y要缩进，表示x，y放在input空间
# #定义两个placeholder，配合上面命名空间，给x，y取个名字
#    x=tf.placeholder(tf.float32,[None,784],name='x-input')#建立一个占位符，None是图片数，784是每幅图的像素个数
#    y=tf.placeholder(tf.float32,[None,10],name='y-input')# 标签，建立一个占位符，10是指0-9十个数
#
# #显示图片
# with tf.name_scope('input_reshape'):#命名空间
#     image_shaped_input=tf.reshape(x,[-1,28,28,1])#把训练或测试的图片显示出来，把x转变成[-1，28，28，1]。-1代表不确定，即上面[None,784]中的None，784转变成[28，28].1代表维度是1，表示黑白，如果是彩色，则是3
#     tf.summary.image('input',image_shaped_input,10)#用summary.image把image_shaped_input存到input里面，一共放10张图片
#
# with tf.name_scope('layer1'):
# #创建一个简单的神经网络，输入层784个神经元，输出层10个神经元，不设隐藏层
#    with tf.name_scope('wights1'):
#       W=tf.Variable(tf.zeros([784,1000]),name='W')#权值，设一个变量，置0
#       variable_summaries(W)#把权值W当作参数，计算的各种指标
#    with tf.name_scope('biases1'):
#       b=tf.Variable(tf.zeros([1000]),name='b')#偏置值
#       variable_summaries(b)#把偏置值b当作参数，计算的各种指标
#    with tf.name_scope('wx_plus_b1'):
#       wx_plus_b=tf.nn.tanh(tf.matmul(x,W)+b)
#
# with tf.name_scope('layer2'):
#      with tf.name_scope('wights2'):
#         W2=tf.Variable(tf.truncated_normal([1000,1000],stddev=0.1))
#      with tf.name_scope('biases2'):
#         b2=tf.Variable(tf.zeros([1000])+0.1)
#      with tf.name_scope('wx_plus_b2'):
#         L2=tf.nn.tanh(tf.matmul(wx_plus_b,W2)+b2)
#
# with tf.name_scope('layer3'):
#      with tf.name_scope('wights3'):
#         W3=tf.Variable(tf.truncated_normal([1000,500],stddev=0.1))
#      with tf.name_scope('biases3'):
#         b3=tf.Variable(tf.zeros([500])+0.1)
#      with tf.name_scope('wx_plus_b3'):
#         L3=tf.nn.tanh(tf.matmul(L2,W3)+b3)
#
# with tf.name_scope('layer4'):
#      with tf.name_scope('wights4'):
#         W4=tf.Variable(tf.truncated_normal([500,10],stddev=0.1))
#      with tf.name_scope('biases4'):
#         b4=tf.Variable(tf.zeros([10])+0.1)
#      with tf.name_scope('softmax'):
#         prediction=tf.nn.softmax(tf.matmul(L3,W4)+b4)#信号总和，经过softmax函数（激活函数）转化成概率值
#
# #使用交叉熵代价函数
# with tf.name_scope('loss'):
#     loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#     tf.summary.scalar('loss',loss)
#
# #使用梯度下降法
# with tf.name_scope('train'):
#     train_step=tf.train.GradientDescentOptimizer(0.2).minimize(loss)
#
# #初始化变量
# init=tf.global_variables_initializer()
#
# with tf.name_scope('accuracy'):
#     with tf.name_scope('correct_prediction'):
# #训练好后求准确率，结果存放在一个布尔型列表中，argmax返回一维张量中最大的值所在的位置
#         correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))#argmax函数是对行或列计算最大值，1表示按行，0表示按列，找到最大概率标签的位置。 equal函数是比较两个参数大小，相等的话返回True，不相等返回False
#     with tf.name_scope('accuracy'):
# #求准确率
#         accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))#cast()是类型转换函数，把布尔型参数转换为32位古典型,然后求平均值。true变成1.0，flse变成0
# #Boolean→数值型：True转换为-1，False转换为0。数值型→Boolean：0转换为False，其他转换为True
#         tf.summary.scalar('accuracy',accuracy)
#
# #产生metadata文件，其实就是test图片对应的label值
# #if tf.gfile.Exists('./projector/metadata.tsv'):
#     #tf.gfile.DeleteRecursively('./projector/metadata.tsv')#j检测是否已有该文件，有的话就将其删除
# with open('./metadata.tsv','w') as f:#生成该文件，并以写的方式打开
#     labels=sess.run(tf.argmax(mnist.test.labels[:],1))#得到test里面的label，argmax找到最大值位置，即label如果是0的话，则数据格式是1000000000；如果是1的话，数据格式是0100000000
#     for i in range(image_num):
#         f.write(str(labels[i])+'\n')#把image_num个图片对应的label写入metadata.tsv文件中
#
# #合并所有的summary,并将其加入到sess.run的语句里
# merged=tf.summary.merge_all()
#
#
# #下面这段是为了生成三维立体动态图像模型，embedding，以及上面各参数的显示
# projector_writer=tf.summary.FileWriter(DIR,sess.graph)#前面是路径，graph存在该文件夹中，从而可以使用tensorboard查看
#
# saver=tf.train.Saver()#保存这个网络的模型
# config=projector.ProjectorConfig()#定义一个配置项
# embed=config.embeddings.add()
# embed.tensor_name=embedding.name#前面embedding变量的名字赋予tensor_name
# embed.metadata_path=DIR+'metadata.tsv'#把metadata.tsv传给embed.metadata_path
# embed.sprite.image_path=DIR+'mnist_10k_sprite.png'#把单张1万个手写数字图片传给embed程序
# embed.sprite.single_image_dim.extend([28,28])#把上面图片切分成每个数字块，即[28，28]的小块
# projector.visualize_embeddings(projector_writer,config)#显示三维图
#
# #训练
# with tf.Session() as sess:
#    sess.run(init)#初始化变量
#    for i in range (max_steps):#一次训练100张图片，总共训练了100*max_step张图片，此方法不去在乎总的数据集迭代了多少次
# #每个批次100个样本
#       batch_xs,batch_ys=mnist.train.next_batch(100)
#       run_options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)#固定写法
#       run_metadata=tf.RunMetadata()#固定写法
#       summary,_=sess.run([merged,train_step],feed_dict={x:batch_xs,y:batch_ys},options=run_options,run_metadata=run_metadata)#每tain训练一次，统计一次参数merged，运行后得到的merged存在summary里。后面两参数固定写法
#       projector_writer.add_run_metadata(run_metadata,'step%03d'%i)#把run_metadata写入tensorboard文件
#       projector_writer.add_summary(summary,i)#将summary和运行周期i写入tensorboard文件
#
#       if i%100==0:#每训练100次，打印一次准确率
#          acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
#          print('Iter'+str(i)+',Testing Accuracy='+str(acc))
#
#    saver.save(sess,DIR+'a_model.ckpt',global_step=max_steps)#训练完了，将训练好的模型保存在此路径
#    projector_writer.close()
#    sess.close()