"""Factory module for motion generation on Human3.6M using MT-VAE."""

import tensorflow as tf

from factory_utils import *
from h36m_seq_utils import get_seq_encoding_model, \
  get_seq_decoding_model, get_cycle_decoding_model
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


def get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse):
  """Factory function to retrieve encoder network model."""
  latent_encoder = _get_network(Z_ENC_FN)
  ############
  ## Latent ##
  ############
  with tf.variable_scope('latent_enc', reuse=reuse):
    tmp_outputs = latent_encoder(
      outputs['his_style'], outputs['fut_style'],
      use_prior, params, is_training=is_training)
  outputs['latent'] = tmp_outputs['latent']
  outputs['mu_latent'] = tmp_outputs['mu']
  outputs['logs2_latent'] = tmp_outputs['logs2']
  return outputs

def get_latent_decoding_model(inputs, outputs, params, is_training, reuse):
  latent_decoder = _get_network(Z_DEC_FN)
  ############
  ## latent ##
  ############
  with tf.variable_scope('latent_dec', reuse=reuse):
    tmp_outputs = latent_decoder(
      outputs['latent'], outputs['his_content'], outputs['his_style'],
      params, is_training=is_training)
  outputs['dec_embedding'] = tmp_outputs['dec_embedding']
  outputs['dec_style'] = tmp_outputs['new_style']
  return outputs

def get_cycle_model(inputs, outputs, params, is_training, reuse):
  assert is_training 
  latent_encoder = _get_network(Z_ENC_FN)
  latent_decoder = _get_network(Z_DEC_FN)
 
  with tf.variable_scope('latent_enc', reuse=True):
    tmp_outputs = latent_encoder(
      tf.zeros_like(outputs['his_style']),
      tf.zeros_like(outputs['fut_style']), True, params, is_training=is_training)
  outputs['cycle_latent'] = tmp_outputs['latent']

  with tf.variable_scope('latent_dec', reuse=True):
    tmp_outputs = latent_decoder(
      outputs['cycle_latent'], outputs['his_content'], outputs['his_style'],
      params, is_training=is_training)
  outputs['cycle_style'] = tmp_outputs['new_style']
  outputs['cycle_dec_embedding'] = tmp_outputs['dec_embedding']

  with tf.variable_scope('latent_enc', reuse=True):
    tmp_outputs = latent_encoder(
      outputs['his_style'], outputs['cycle_style'],
      False, params, is_training=is_training)
  outputs['cycle_mu_latent'] = tmp_outputs['mu']
  outputs['cycle_logs2_latent'] = tmp_outputs['logs2']
  return outputs

def get_model_fn(params, is_training=False,
                       use_prior=False, reuse=False):
  def model(inputs):
    outputs = get_seq_encoding_model(inputs, params, is_training, reuse)
    outputs = get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse)
    outputs = get_latent_decoding_model(inputs, outputs, params, is_training, reuse)
    outputs = get_seq_decoding_model(inputs, outputs, params, is_training, reuse)
    if is_training and hasattr(params, 'cycle_model') and params.cycle_model:
      outputs = get_cycle_model(inputs, outputs, params, is_training, reuse)
      outputs = get_cycle_decoding_model(inputs, outputs, params, is_training, reuse)
    return outputs
  return model

def get_sample_fn(params, is_training=False,
                  use_prior=False, reuse=False, output_length=None):
  """Factory function to retrieve a network model for iterative sampling."""
  
  def model(inputs):
    outputs = get_seq_encoding_model(inputs, params, is_training, reuse)
    outputs = get_latent_encoding_model(inputs, outputs, params, is_training, use_prior, reuse)
    outputs = get_latent_decoding_model(inputs, outputs, params, is_training, reuse)
    outputs = get_seq_decoding_model(inputs, outputs, params, is_training, reuse, output_length)
    return outputs    
  return model

