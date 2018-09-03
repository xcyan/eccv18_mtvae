"""Factory module for Human3.6M Keypoint Sequence Analogy-making using MT-VAE."""

import tensorflow as tf

from factory_utils import get_network, get_rnn_cell
from h36m_analogy_utils import analogy_seq_encoding_model, analogy_seq_decoding_model
import mtvae_latent_encoder
import mtvae_latent_decoder
import numpy as np
import random

Z_ENC_FN = 'mtvae_latent_encoder'
Z_DEC_FN = 'mtvae_latent_decoder'

NAME_TO_NETS = {
  Z_ENC_FN: mtvae_latent_encoder,
  Z_DEC_FN: mtvae_latent_decoder,
}

def _get_network(name):
  return get_network(name, NAME_TO_NETS)

def analogy_making_model(inputs, params, is_training, reuse, output_length=None):
  """Factory function to retrieve analogy-making model."""
  latent_encoder = _get_network(Z_ENC_FN)
  latent_decoder = _get_network(Z_DEC_FN)

  outputs = analogy_seq_encoding_model(inputs, params, is_training, reuse)

  with tf.variable_scope('latent_enc', reuse=reuse):
    tmp_outputs = latent_encoder(
      outputs['A_style'], outputs['B_style'],
      False, params, is_training=is_training)
  outputs['AtoB_latent'] = tmp_outputs['latent']

  ##############################
  ## Compute f*(D) = f(C) + T ##
  ##############################
  outputs['CtoD_latent'] = outputs['AtoB_latent']
  with tf.variable_scope('latent_dec', reuse=reuse):
    tmp_outputs = latent_decoder(
      outputs['CtoD_latent'], outputs['C_content'], outputs['C_style'],
      params, is_training=is_training)
  outputs['dec_D_embedding'] = tmp_outputs['dec_embedding']
  outputs['D_style'] = tmp_outputs['new_style']

  outputs = analogy_seq_decoding_model(inputs, outputs, params,
    is_training, reuse, output_length)
  #
  return outputs

def get_model_fn(params, is_training=False, reuse=False, output_length=None):
  """Factory function to retrieve a network model for analogy-making."""
  def model(inputs):
    outputs = analogy_making_model(inputs, params, is_training, reuse, output_length)
    return outputs
  return model 

