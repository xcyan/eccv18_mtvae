"""Latent decoder for MTVAE."""

import tensorflow as tf
from layer_utils import layer_norm, resnet_block

slim = tf.contrib.slim

def concat_fn(net, prev_input, params):
  out_dim = net.get_shape().as_list()[1]
  net = tf.concat([net, prev_input], axis=1)
  net = slim.fully_connected(
    net, out_dim, activation_fn=tf.nn.relu, normalizer_fn=layer_norm)
  net = slim.fully_connected(
    net, out_dim, activation_fn=None, normalizer_fn=None)
  return net

def add_fn(net, prev_input, params):
  out_dim = net.get_shape().as_list()[1]
  net = tf.concat([net, prev_input], axis=1)
  for layer_i in xrange(params.latent_fc_layers):
    net = resnet_block(net, out_dim, afn=tf.nn.relu, nfn=None)
    net = layer_norm(net)
  net = slim.fully_connected(
    net, out_dim, activation_fn=None, normalizer_fn=None)
  return net

# TODO(xcyan): verify concat_fn.
def get_interaction_fn(mode):
  if mode == 'add':
    return add_fn
  elif mode == 'concat':
    return concat_fn

def model(latent_input, prev_content, prev_style, params, is_training):
  embed_dim = params.embed_dim
  state_size = params.dec_rnn_size * 2
  style_dim = params.embed_dim - params.content_dim

  outputs = dict()
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    net = latent_input
    inv_z = slim.fully_connected(
      net, style_dim, activation_fn=None,
      normalizer_fn=layer_norm)
    interaction_fn = get_interaction_fn(params.dec_interaction)
    slim.summaries.add_histogram_summary(
      inv_z, name='delta_z_activation', prefix='summaries')
    T_new = interaction_fn(inv_z * params.use_latent, prev_style, params)
    if hasattr(params, 'T_layer_norm') and params.T_layer_norm > 0:
      T_new = layer_norm(T_new)
    new_style = T_new + prev_style
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
