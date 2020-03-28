import tensorflow as tf
import tensorflow.contrib.slim as slim

from model.model_fn import build_model
from utils.train_utils import Params


def display_model(ckpt_path):
    with tf.Session() as sess:
        for var_name, value in tf.contrib.framework.list_variables(ckpt_path):
            if len(var_name.split('/')) > 2:
                print(var_name, ": ", value)
            #     var_name_reloaded = rename_dict[var_name]
            #     var_org = tf.contrib.framework.load_variable(args.org_checkpoint_path, var_name)
            #     var_reloaded = tf.contrib.framework.load_variable(args.checkpoint_path, var_name_reloaded)
            #     print(var_name, '||', var_name_reloaded)
            #     print(var_org - var_reloaded)

            # if len(var_name.split('/')) > 2:
            #     if var_name.split('/')[3] == 'gamma':
            #         var = tf.contrib.framework.load_variable(args.checkpoint_path, var_name)
            #         print(var)


def get_variables_available_in_checkpoint(variables,
                                          checkpoint_path,
                                          include_global_step=True):
  """Returns the subset of variables available in the checkpoint.

  Inspects given checkpoint and returns the subset of variables that are
  available in it.

  TODO(rathodv): force input and output to be a dictionary.

  Args:
    variables: a list or dictionary of variables to find in checkpoint.
    checkpoint_path: path to the checkpoint to restore variables from.
    include_global_step: whether to include `global_step` variable, if it
      exists. Default True.

  Returns:
    A list or dictionary of variables.
  Raises:
    ValueError: if `variables` is not a list or dict.
  """
  if isinstance(variables, list):
    variable_names_map = {}
    for variable in variables:
      if isinstance(variable, tf_variables.PartitionedVariable):
        name = variable.name
      else:
        name = variable.op.name
      variable_names_map[name] = variable
  elif isinstance(variables, dict):
    variable_names_map = variables
  else:
    raise ValueError('`variables` is expected to be a list or dict.')
  ckpt_reader = tf.train.NewCheckpointReader(checkpoint_path)
  ckpt_vars_to_shape_map = ckpt_reader.get_variable_to_shape_map()
  if not include_global_step:
    ckpt_vars_to_shape_map.pop(tf.GraphKeys.GLOBAL_STEP, None)
  vars_in_ckpt = {}
  for variable_name, variable in sorted(variable_names_map.items()):
    if variable_name in ckpt_vars_to_shape_map:
      if ckpt_vars_to_shape_map[variable_name] == variable.shape.as_list():
        vars_in_ckpt[variable_name] = variable
      else:
        logging.warning('Variable [%s] is available in checkpoint, but has an '
                        'incompatible shape with model variable. Checkpoint '
                        'shape: [%s], model variable shape: [%s]. This '
                        'variable will not be initialized from the checkpoint.',
                        variable_name, ckpt_vars_to_shape_map[variable_name],
                        variable.shape.as_list())
    else:
      logging.warning('Variable [%s] is not available in checkpoint',
                      variable_name)
  if isinstance(variables, list):
    return list(vars_in_ckpt.values())
  return vars_in_ckpt

def variables_to_restore(ckpt_path):
    params = Params('../model/parameters.json')
    model = build_model(params)
    ckpt_path = tf.train.latest_checkpoint(ckpt_path)
    # print(ckpt_path)
    variables_to_restore = slim.get_variables_to_restore(include=['model'])
    print(ckpt_path)
    print([v.name for v in variables_to_restore])


if __name__ == '__main__':

    # display_model('D:/Pycharm/Projects/Triplet-Loss-Tensorflow/checkpoints/frozen_graph/model.ckpt')
    variables_to_restore('D:\\Pycharm\\Projects\\Triplet-Loss-Tensorflow\\checkpoints')