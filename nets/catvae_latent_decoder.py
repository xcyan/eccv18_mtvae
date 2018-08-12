"""Latent decoder for MT-VAE (concat) or Vanilla VAE."""

import tensorflow as tf
from layer_utils import layer_norm, resnet_block

slim = tf.contrib.slim

def model(latent_input, prev_content, prev_style, params, is_training):
  embed_dim = params.embed_dim
  state_size = params.dec_rnn_size * 2
  style_dim = params.embed_dim - params.content_dim
  # 
  outputs = dict()
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    net = latent_input
    inv_z = slim.fully_connected(
      net, style_dim, activation_fn=None,
      normalizer_fn=layer_norm)
    net = tf.concat([inv_z, prev_style], 1)
    for layer_i in xrange(params.latent_fc_layers):
      net = resnet_block(net, style_dim, afn=tf.nn.relu, nfn=None)
      net = layer_norm(net)
    new_style = slim.fully_connected(
      net, style_dim, activation_fn=None, normalizer_fn=None)
    if hasattr(params, 'T_layer_norm') and params.T_layer_norm > 0:
      new_style = layer_norm(new_style)
    outputs['new_style'] = tf.identity(new_style)
    #  
    net = tf.concat([prev_content, new_style], axis=1)
    for layer_i in xrange(params.dec_fc_layers):
      net = resnet_block(net, embed_dim, afn=tf.nn.relu, nfn=None)
      net = layer_norm(net)
    #
    h0 = slim.fully_connected(
      net, state_size/2, activation_fn=tf.nn.tanh)
    c0 = slim.fully_connected(
      net, state_size/2, activation_fn=None)
  outputs['dec_embedding'] = tf.concat([c0, h0], 1)
  return outputs
