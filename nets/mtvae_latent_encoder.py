"""Latent encoder for MT-VAE."""

import tensorflow as tf
import numpy as np
import rnn

from layer_utils import layer_norm, resnet_block

def model(his_style, fut_style, use_prior, params, is_training):
  quantity = his_style.get_shape().as_list()[0]
  embed_dim = params.embed_dim
  noise_dim = params.noise_dim

  outputs = dict()
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    slim.summaries.add_histogram_summary(
      his_style, name='his_style_activation', prefix='summaries')
    slim.summaries.add_histogram_summary(
      fut_style, name='fut_style_activation', prefix='summaries')
    net = layer_norm(fut_style - his_style)
    slim.summaries.add_histogram_summary(
      net, name='fut_minus_his_activation', prefix='summaries')
    for layer_i in xrange(params.latent_fc_layers):
      net = resnet_block(net, embed_dim, afn=tf.nn.relu, nfn=None)
      net = layer_norm(net)
    #
    mu = slim.fully_connected(net, noise_dim, activation_fn=None)
    logs2 = slim.fully_connected(net, noise_dim, activation_fn=None)
  outputs['mu'] = mu
  outputs['logs2'] = logs2

  if is_training or (not hasattr(params, 'sample_pdf')) or (not hasattr(params, 'sample_temp')):
    temp = 1.0
  else:
    temp = params.sample_temp

  noise_vec = tf.random_normal(shape=[quantity, noise_dim], dtype=tf.float32)
  if use_prior:
    outputs['latent'] = noise_vec * temp
  else:
    outputs['latent'] = mu + tf.multiply(noise_vec, tf.exp(0.5 * logs2)) * temp
  return outputs

