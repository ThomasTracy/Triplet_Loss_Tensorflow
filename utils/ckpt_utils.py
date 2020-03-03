import tensorflow as tf


def display_model(ckpt_path):
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(ckpt_path):
            if len(var_name.split('/')) > 2:
                print(var_name)
            #     var_name_reloaded = rename_dict[var_name]
            #     var_org = tf.contrib.framework.load_variable(args.org_checkpoint_path, var_name)
            #     var_reloaded = tf.contrib.framework.load_variable(args.checkpoint_path, var_name_reloaded)
            #     print(var_name, '||', var_name_reloaded)
            #     print(var_org - var_reloaded)

            # if len(var_name.split('/')) > 2:
            #     if var_name.split('/')[3] == 'gamma':
            #         var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)
            #         print(var)

if __name__ == '__main__':
    display_model('D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints/frozen_graph/model.ckpt')