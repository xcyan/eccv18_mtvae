"""Helper layers."""

import tensorflow as tf

slim = tf.contrib.slim
layer_norm_fn = tf.contrib.layers.layer_norm


def layer_norm(x):
  return layer_norm_fn(x, center=False, scale=False)


def resnet_block(inputs, fc_dim, afn, nfn):
  shortcut = slim.fully_connected(
    inputs, fc_dim, activation_fn=afn, normalizer_fn=nfn)
  deep1 = slim.fully_connected(
    inputs, int(fc_dim/2), activation_fn=afn, normalizer_fn=nfn)
  deep2 = slim.fully_connected(
    deep1, int(fc_dim/2), activation_fn=afn, normalizer_fn=nfn)
  deep3 = slim.fully_connected(
    deep2, fc_dim, activation_fn=afn, normalizer_fn=nfn)
  outputs = deep3 + shortcut
  return outputs
