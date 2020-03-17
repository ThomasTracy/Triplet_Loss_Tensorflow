'''
Functions to export ckpt file to pb file
'''
import tensorflow as tf
import logging
import os

from model.model_fn import build_model
from utils.train_utils import Params
from model.triplet_loss import batch_all_center_triplet_loss

from tensorflow.python.training import saver as saver_lib
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.saved_model import signature_constants

slim = tf.contrib.slim

REFERENCE_SIZE = 51

def _image_tensor_input_placeholder(input_shape=None):
    if input_shape is None:
        input_shape = (None, None, None, 3)
    input_tensor = tf.placeholder(
        dtype=tf.float32, shape=input_shape, name='image_input'
    )
    return input_tensor, input_tensor


def _add_output_tensor_nodes(postprocessed_tensors,
                             output_collection_name='inference_op'):
    """Adds output nodes.

        Adjust according to specified implementations.

        Adds the following nodes for output tensors:
            * classes: A float32 tensor of shape [batch_size] containing
            predicted classes of inputs.

        Args:
            postprocessed_tensors: A dictionary containing the following fields:
                'classes': [batch_size].
            output_collection_name: Name of collection to add output tensors to.

        Returns:
            A tensor dict containing the added output tensor nodes.
        """
    outputs = {}
    classes = postprocessed_tensors.get('classes')
    outputs['classes'] = tf.identity(classes, name='classes')
    for key in outputs:
        tf.add_to_collection(output_collection_name, outputs[key])

    return outputs


def _get_outputs_from_inputs(input_tensor, model, output_collection_name, params):
    # params = Params('../model/parameters.json')

    inputs = tf.to_float(input_tensor)
    outputs = model(inputs)
    # batch_size = inputs.shape[0]
    # batch_size = batch_size - REFERENCE_SIZE
    # images_input = inputs[:batch_size]
    # images_ref = inputs[batch_size:]
    outputs = batch_all_center_triplet_loss(params, outputs)
    outputs = tf.argmin(outputs, axis=1)
    postprecessed_outputs = {'classes': outputs}

    return _add_output_tensor_nodes(postprecessed_outputs, output_collection_name)


def _build_model_graph(model, input_shape, output_collection_name, params):
    placeholder_tensor, input_tensor = _image_tensor_input_placeholder(input_shape)
    outputs = _get_outputs_from_inputs(input_tensor, model, output_collection_name, params)

    slim.get_or_create_global_step()

    return outputs, placeholder_tensor


def write_graph_and_checkpoint(inference_graph_def,
                               model_path,
                               input_saver_def,
                               trained_checkpoint_prefix):
    for node in inference_graph_def.node:
        node.device = ''
    with tf.Graph().as_default():
        tf.import_graph_def(inference_graph_def, name='')
        with tf.Session() as sess:
            saver = saver_lib.Saver(saver_def=input_saver_def,
                                    save_relative_paths=True)
            saver.restore(sess, trained_checkpoint_prefix)
            saver.save(sess, model_path)


def freeze_graph_with_def_protos(
        input_graph_def,
        input_saver_def,
        input_checkpoint,
        output_node_names,
        restore_op_names,
        filename_tensor_name,
        clear_devices,
        initializer_nodes,
        variable_names_blacklist=''):
    if not saver_lib.checkpoint_exists(input_checkpoint):
        raise ValueError("Checkpoint {} doesnt exist".format(input_checkpoint))
    if not output_node_names:
        raise ValueError("Output name cant be None")
    if clear_devices:
        for node in input_graph_def.node:
            node.device = ''

    with tf.Graph().as_default():
        tf.import_graph_def(input_graph_def, name='')
        config = tf.ConfigProto(graph_options=tf.GraphOptions())
        with tf.Session(config=config) as sess:
            if input_saver_def:
                saver = saver_lib.Saver(saver_def=input_saver_def)
                saver.restore(sess, input_checkpoint)

            variable_names_blacklist = (variable_names_blacklist.split(',') if
                                        variable_names_blacklist else None)
            output_graph_def = graph_util.convert_variables_to_constants(
                sess,
                input_graph_def,
                output_node_names.split(','),
                variable_names_blacklist=variable_names_blacklist
            )

    return output_graph_def


def write_frozen_graph(frozen_graph_path, frozen_graph_def):

    with gfile.GFile(frozen_graph_path, 'wb') as f:
        f.write(frozen_graph_def.SerializeToString())

    logging.info('%d ops in the final graph.', len(frozen_graph_def.node))


def write_saved_model(saved_model_path,
                      frozen_graph_def,
                      inputs,
                      outputs):
    with tf.Graph().as_default():
        with tf.Session() as sess:
            tf.import_graph_def(frozen_graph_def, name='')

            builder = tf.saved_model.builder.SavedModelBuilder(
                saved_model_path
            )

            tensor_info_inputs = {
                'inputs': tf.saved_model.utils.build_tensor_info(inputs)
            }
            tensor_info_outputs = {}
            for k, v in outputs.items():
                tensor_info_outputs[k] = tf.saved_model.utils.build_tensor_info(
                    v)

            detection_signature = (
                tf.saved_model.signature_def_utils.build_signature_def(
                    inputs=tensor_info_inputs,
                    outputs=tensor_info_outputs,
                    method_name=signature_constants.PREDICT_METHOD_NAME))

            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                signature_def_map={
                    signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                        detection_signature,
                },
            )
            builder.save()


def export_inference_graph(model,
                           trained_checkpoint_prefix,
                           output_dictionary,
                           params,
                           input_shape=None,
                           output_collection_name='inference_op'):
    """Exports inference graph for the desired graph.

        Args:
            model: A model defined by model.py.
            trained_checkpoint_prefix: Path to the trained checkpoint file.
            output_directory: Path to write outputs.
            input_shape: Sets a fixed shape for an 'image_tensor' input. If not
                specified, will default to [None, None, None, 3].
            output_collection_name: Name of collection to add output tensors to.
                If None, does not add output tensors to a collection.
        """
    if not os.path.exists(output_dictionary):
        os.mkdir(output_dictionary)
    frozen_graph_path = os.path.join(output_dictionary,
                                     'frozen_inference_graph.pb')
    save_model_path = os.path.join(output_dictionary, 'saved_model')
    model_path = os.path.join(output_dictionary, 'model.ckpt')

    outputs, placeholder_tensor = _build_model_graph(model, input_shape, output_collection_name, params)

    saver = tf.train.Saver()
    input_saver_def = saver.as_saver_def()

    write_graph_and_checkpoint(
        inference_graph_def=tf.get_default_graph().as_graph_def(),
        model_path=model_path,
        input_saver_def=input_saver_def,
        trained_checkpoint_prefix=trained_checkpoint_prefix
    )

    output_node_names = ','.join(outputs.keys())
    frozen_graph_def = freeze_graph_with_def_protos(
        input_graph_def=tf.get_default_graph().as_graph_def(),
        input_saver_def=input_saver_def,
        input_checkpoint=trained_checkpoint_prefix,
        output_node_names=output_node_names,
        restore_op_names='save/restore_all',
        filename_tensor_name='save/Const:0',
        clear_devices=True,
        initializer_nodes=''
    )
    write_frozen_graph(frozen_graph_path, frozen_graph_def)
    write_saved_model(save_model_path, frozen_graph_def,
                      placeholder_tensor, outputs)


if __name__ == '__main__':
    trained_checkpoint_prefix = 'D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints\\model.ckpt-3933'
    output_dictionary = 'D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints\\frozen_graph_full'
    params = Params('../model/parameters.json')
    with tf.variable_scope('model'):
        model = build_model(params)
    input_shape = [None, params.image_size, params.image_size, 3]
    export_inference_graph(model,
                           trained_checkpoint_prefix,
                           output_dictionary,
                           params,
                           input_shape)