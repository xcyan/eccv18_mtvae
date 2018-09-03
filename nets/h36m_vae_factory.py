"""Factory module for motion generation on Human3.6M using Vanilla VAE."""

import tensorflow as tf

from factory_utils import get_network, get_rnn_cell
from h36m_seq_utils import get_singleseq_encoding_model
from h36m_seq_utils import get_seq_decoding_model
import catvae_latent_encoder
import catvae_latent_decoder
import numpy as np
import random

Z_ENC_FN = 'catvae_latent_encoder'
Z_DEC_FN = 'catvae_latent_decoder'

NAME_TO_NETS = {
  Z_ENC_FN: catvae_latent_encoder,
  Z_DEC_FN: catvae_latent_decoder,
}

def _get_network(name):
  return get_network(name, NAME_TO_NETS)

def get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse):
  latent_encoder = _get_network(Z_ENC_FN)
  # latent
  with tf.variable_scope('latent_enc', reuse=reuse):
    tmp_outputs = latent_encoder(
      tf.zeros_like(outputs['his_style']), outputs['fut_style'],
      use_prior, params, is_training=is_training)
  outputs['latent'] = tmp_outputs['latent']
  outputs['mu_latent'] = tmp_outputs['mu']
  outputs['logs2_latent'] = tmp_outputs['logs2']
  return outputs

def get_latent_decoding_model(inputs, outputs, params, is_training, reuse):
  latent_decoder = _get_network(Z_DEC_FN)
  #
  with tf.variable_scope('latent_dec', reuse=reuse):
    tmp_outputs = latent_decoder(
      outputs['latent'], outputs['his_content'], outputs['his_style'],
      params, is_training=is_training)
  outputs['dec_embedding'] = tmp_outputs['dec_embedding']
  outputs['dec_style'] = tmp_outputs['new_style']
  return outputs

def get_model_fn(params, is_training=False, use_prior=False, reuse=False):
  def model(inputs):
    outputs = get_singleseq_encoding_model(inputs, params, is_training, reuse)
    outputs = get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse)
    outputs = get_latent_decoding_model(inputs, outputs, params, is_training, reuse)
    outputs = get_seq_decoding_model(inputs, outputs, params, is_training, reuse)
    return outputs
  return model

def get_sample_fn(params, is_training=False,
                  use_prior=False, reuse=False, output_length=None):
  """Factory function to retrieve a network model for iterative sampling."""
  
  def model(inputs):
    outputs = get_singleseq_encoding_model(inputs, params, is_training, reuse)
    outputs = get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse)
    outputs = get_latent_decoding_model(inputs, outputs, params, is_training, reuse) 
    outputs = get_seq_decoding_model(inputs, outputs, params, is_training, reuse, output_length)
    return outputs
  return model

