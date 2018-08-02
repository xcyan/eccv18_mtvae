"""Architecture file for Human3.6M landmark encoding network."""

import tensorflow as tf
import numpy as np

from layer_utils import layer_norm
from layer_utils import resnet_block
slim = tf.contrib.slim

def model(input_keypoint, seq_len, rnn_cell_fw, rnn_cell_bw, params, is_training):
  shp = input_keypoint.get_shape().as_list()
  quantity, max_length = shp[0], shp[1]
  embed_dim = params.embed_dim
  keypoint_dim = params.keypoint_dim
  
  input_keypoint = input_keypoint * 2 - 1
  #  
  net = tf.reshape(input_keypoint, [quantity, max_length, keypoint_dim*2])
  outputs = dict()
  outputs['features'] = net

  if seq_len is None:
    return outputs

  if params.use_bidirection_lstm:
    _, states = tf.nn.bidirectional_dynamic_rnn(
      rnn_cell_fw,
      rnn_cell_bw,
      outputs['features'],
      dtype=tf.float32,
      sequence_length=seq_len,
      swap_memory=True,
      scope='ENC_RNN')

    hid_state = tf.concat([
      rnn_cell_fw.get_output(states[0]),
      rnn_cell_bw.get_output(states[1])], axis=1)
    mem_state = tf.concat([
      rnn_cell_fw.get_memory(states[0]),
      rnn_cell_bw.get_memory(states[1])], axis=1)
  else:
    _, states = tf.nn.dynamic_rnn(
      rnn_cell_fw,
      outputs['features'],
      dtype=tf.float32,
      sequence_length=seq_len,
      swap_memory=True,
      scope='ENC_RNN')
    hid_state = rnn_cell_fw.get_output(states)
    mem_state = rnn_cell_fw.get_memory(states)
  
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

