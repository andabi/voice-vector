# !/usr/bin/env python2.7
"""Export model given existing training checkpoints.

The model is exported as SavedModel with proper signatures that can be loaded by
standard tensorflow_model_server.
"""

from __future__ import print_function

import sys
sys.path.append('.')
sys.path.append('..')

import argparse
import os.path

import tensorflow as tf

from data_load import VoxCelebMeta
from hparam import hparam as hp
from model import ClassificationModel
from tensorpack.input_source.input_source import PlaceholderInput
from tensorpack.tfutils import TowerContext


def export(output_dir, ckpt=None, model_version=1):
  # Define model.
  audio_meta_train = VoxCelebMeta(hp.train.data_path, hp.train.meta_path)
  model = ClassificationModel(num_classes=audio_meta_train.num_speaker, **hp.model)

  with TowerContext('', is_training=False):
    input = PlaceholderInput()
    input.setup(model.get_inputs_desc())
    model.build_graph(*input.get_input_tensors())

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # Restore variables from training checkpoints.
    ckpt = ckpt if ckpt else tf.train.latest_checkpoint(hp.logdir)
    if ckpt:
      tf.train.Saver().restore(sess, ckpt)
      print('Successfully loaded model: {} from {}'.format(ckpt, ckpt))
    else:
      print('No checkpoint file found at {}'.format(ckpt))
      return

    # Export inference model.
    output_path = os.path.join(
      tf.compat.as_bytes(output_dir),
      tf.compat.as_bytes(str(model_version)))
    print('Exporting trained model to', output_path)
    builder = tf.saved_model.builder.SavedModelBuilder(output_path)

    # Build the signature_def_map.
    inputs_tensor_info = tf.saved_model.utils.build_tensor_info(model.x)
    prob_output_tensor_info = tf.saved_model.utils.build_tensor_info(model.prob)
    embedding_output_tensor_info = tf.saved_model.utils.build_tensor_info(model.y)

    predict_signature = (
      tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'x': inputs_tensor_info},
        outputs={
          'prob': prob_output_tensor_info,
          'embedding': embedding_output_tensor_info,
        },
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
      ))

    builder.add_meta_graph_and_variables(
      sess, [tf.saved_model.tag_constants.SERVING],
      signature_def_map={
        'predict': predict_signature
      })

    builder.save()
    print('Successfully exported model to %s' % output_dir)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('case', type=str, help='experiment case name.')
  parser.add_argument('output_dir', type=str, help='directory where to export model.')
  parser.add_argument('-ckpt', help='checkpoint to load model.')
  parser.add_argument('-model_version', default=1, help='version number of the model.')
  args = parser.parse_args()

  hp.set_hparam_yaml(args.case)

  export(output_dir=args.output_dir, ckpt=args.ckpt, model_version=args.model_version)