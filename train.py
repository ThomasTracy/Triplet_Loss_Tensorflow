import argparse
import os
import tensorflow as tf

from model.model_fn import model_fn
from model.input_fn import train_input_fn
from utils.train_utils import Params


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='D:\Pycharm\Projects\Triplet-Loss-Tensorflow\model',
                    help="Experiment directory containing params.json")
parser.add_argument('--data_dir', default='data/mnist',
                    help="Directory containing the dataset")
parser.add_argument('--checkpoints', default='D:\Pycharm\Projects\Triplet-Loss-Tensorflow\checkpoints',
                    help="Directory containing the checkpoints")

if __name__ == '__main__':
    tf.reset_default_graph()
    tf.logging.set_verbosity(tf.logging.INFO)

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'parameters.json')
    assert os.path.isfile(json_path), "File {} does not exist".format(json_path)
    params = Params(json_path)

    # Build model
    tf.logging.info("----------------------------------------------------------------------")
    tf.logging.info("                          Creating model")
    tf.logging.info("----------------------------------------------------------------------")
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
    # session_config = tf.ConfigProto(gpu_options=gpu_options)
    session_config = tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)
    session_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    config = tf.estimator.RunConfig(
        session_config=session_config,
        tf_random_seed=521,
        model_dir=args.checkpoints,
        save_checkpoints_secs=params.save_checkpoints_secs,
        save_summary_steps=params.save_summary_steps,
        keep_checkpoint_max=params.checkpoints_max
    )
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=params.finetune_path)
    estimator = tf.estimator.Estimator(model_fn=model_fn,
                                       params=params, config=config,
                                       warm_start_from=ws)
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(params),
                                        max_steps=params.total_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(params),
                                      steps=None,
                                      start_delay_secs=120,
                                      throttle_secs=120)

    # Training
    tf.logging.info("----------------------------------------------------------------------")
    tf.logging.info("           Start training for {} epochs".format(params.num_epochs))
    tf.logging.info("----------------------------------------------------------------------")

    # estimator.train(lambda: build_dataset(params))
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)