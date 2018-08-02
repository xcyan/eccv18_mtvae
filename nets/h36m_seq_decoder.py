"""Architecture file for Human3.6M sequence decoding network."""

import tensorflow as tf
import numpy as np
import random

from layer_utils import layer_norm, resnet_block
slim = tf.contrib.slim


def model(prev_states, dec_input, dec_cell, max_length, params, is_training):
  embed_dim = params.embed_dim
  keypoint_dim = params.keypoint_dim
  quantity = prev_states.get_shape().as_list()[0]
  # preprocess.
  empty_input = tf.zeros([quantity, max_length, 1], dtype=tf.float32)
  if dec_input is None:
    rnn_input = empty_input
  else:
    rnn_input = tf.tile(tf.expand_dims(dec_input, 1), [1, max_length, 1])
  rnn_output, states = tf.nn.dynamic_rnn(
    dec_cell,
    rnn_input,
    initial_state=prev_states,
    swap_memory=True,
    dtype=tf.float32,
    scope='DEC_RNN')

  rnn_output = tf.reshape(rnn_output, [quantity * max_length, params.dec_rnn_size])
  outputs = dict()
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    net = rnn_output
    sample_keypoint = slim.fully_connected(
      net, keypoint_dim*2, activation_fn=tf.nn.sigmoid, scope='final')
  
  #####################
  ## Reshape Tensors ##
  #####################
  sample_keypoint = tf.reshape(sample_keypoint, [quantity, max_length, keypoint_dim, 2])
  outputs['keypoint_output'] = sample_keypoint
  outputs['states'] = states
  return outputs
