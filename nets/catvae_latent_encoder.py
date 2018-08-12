"""Architectur file for MT-VAE (concat) and Vanilla VAE."""

import tensorflow as tf
import numpy as np

from layer_utils import layer_norm
from layer_utils import resnet_block
slim = tf.contrib.slim

def model(his_style, fut_style, use_prior, params, is_training):
  quantity = his_style.get_shape().as_list()[0]
  embed_dim = params.embed_dim
  noise_dim = params.noise_dim

  outputs = dict()
  with slim.arg_scope(
    [slim.fully_connected],
      weights_initializer=tf.truncated_normal_initializer(stddev=0.002, seed=1)):
    # Vector concat
    net = tf.concat([his_style, fut_style], axis=1)
    for i in xrange(params.latent_fc_layers):
      net = resnet_block(net, embed_dim, afn=tf.nn.relu, nfn=None)
      net = layer_norm(net)
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

