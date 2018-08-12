"""Architecture file for Single Stream Sequence Encoder (for Vanilla VAE) on Human3.6M."""

import tensorflow as tf
import numpy as np

from layer_utils import layer_norm, resnet_block
slim = tf.contrib.slim


def model(prev_state, input_keypoint, seq_len, rnn_cell_fw, params, is_training):
  shp = input_keypoint.get_shape().as_list()
  quantity, max_length = shp[0], shp[1]
  embed_dim = params.embed_dim
  keypoint_dim = params.keypoint_dim
  
  input_keypoint = input_keypoint * 2 - 1
  #  
  input_keypoint = tf.reshape(input_keypoint, [quantity, max_length, keypoint_dim*2])
  outputs = dict()

  if seq_len is None:
    return outputs
  
  assert (not params.use_bidirection_lstm)
  _, states = tf.nn.dynamic_rnn(
    rnn_cell_fw,
    input_keypoint,
    initial_state=prev_state,
    sequence_length=seq_len,
    swap_memory=True,
    dtype=tf.float32,
    scope='ENC_RNN')
  hid_state = rnn_cell_fw.get_output(states)
  outputs['states'] = states
   
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    hid_state = slim.fully_connected(
      hid_state, embed_dim, activation_fn=None, normalizer_fn=None, 
      scope='embedding')
  outputs['hid_embedding'] = hid_state
  if hasattr(params, 'content_dim'):
    outputs['content'] = hid_state[:, 0:params.content_dim]
    outputs['style'] = hid_state[:, params.content_dim:]
    if hasattr(params, 'T_layer_norm') and params.T_layer_norm > 0:
      outputs['style'] = layer_norm(outputs['style'])
  return outputs
