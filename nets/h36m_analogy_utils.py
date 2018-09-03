"""Factory module for H36M keypoint sequence analogy-making."""

import tensorflow as tf

from factory_utils import get_network, get_rnn_cell
import h36m_seq_encoder as keypoint_seq_encoder
import h36m_seq_decoder as keypoint_seq_decoder
import h36m_singleseq_encoder as keypoint_singleseq_encoder
import rnn
import numpy as np
import random

ENC_FN = 'keypoint_seq_encoder'
FUT_DEC_FN = 'keypoint_seq_decoder'
SINGLESEQ_ENC_FN = 'keypoint_singleseq_encoder'

NAME_TO_NETS = {
  ENC_FN: keypoint_seq_encoder,
  FUT_DEC_FN: keypoint_seq_decoder,
  SINGLESEQ_ENC_FN: keypoint_singleseq_encoder,
}

NAME_TO_RNNCELL = {
  'lstm': rnn.LSTMCell,
  'ln_lstm': rnn.LayerNormLSTMCell,
}

def _get_network(name):
  return get_network(name, NAME_TO_NETS)


def analogy_singleseq_encoding_model(inputs, params, is_training, reuse):
  enc_cell_fn = NAME_TO_RNNCELL[params.enc_model]
  recurrent_dropout_prob = 1.0
  if is_training:
    recurrent_dropout_prob = params.recurrent_dropout_prob

  assert (not params.use_bidirection_lstm)
  enc_cell = get_rnn_cell(
    enc_cell_fn, params.enc_rnn_size,
    use_dropout=is_training and params.use_recurrent_dropout,
    keep_prob=recurrent_dropout_prob, is_bidir=False)
  singleseq_encoder = _get_network(SINGLESEQ_ENC_FN)

  outputs = dict()
  ##############################
  ## Encoding T = f(b) - f(a) ##
  ##############################
  with tf.variable_scope('seq_enc', reuse=reuse):
    tmp_outputs = singleseq_encoder(
      None, inputs['A_landmarks'], inputs['A_lens'],
      enc_cell, params, is_training=is_training)
  enc_state = tmp_outputs['states']
  if hasattr(params, 'content_dim'):
    outputs['A_content'] = tmp_outputs['content']
    outputs['A_style'] = tmp_outputs['style']

  with tf.variable_scope('seq_enc', reuse=True):
    tmp_outputs = singleseq_encoder(
      enc_state, inputs['B_landmarks'], inputs['B_lens'],
      enc_cell, params, is_training=is_training)
  if hasattr(params, 'content_dim'):
    outputs['B_content'] = tmp_outputs['content']
    outputs['B_style'] = tmp_outputs['style']

  with tf.variable_scope('seq_enc', reuse=True):
   tmp_outputs = singleseq_encoder(
      None, inputs['C_landmarks'], inputs['C_lens'],
      enc_cell, params, is_training=is_training)
  outputs['C_enc_state'] = tmp_outputs['states']
  if hasattr(params, 'content_dim'):
    outputs['C_content'] = tmp_outputs['content']
    outputs['C_style'] = tmp_outputs['style']
  return outputs

def analogy_seq_encoding_model(inputs, params, is_training, reuse): 
  """Factory function to retrieve analogy-making model."""
  enc_cell_fn = NAME_TO_RNNCELL[params.enc_model]
  recurrent_dropout_prob = 1.0
  if is_training:
    recurrent_dropout_prob = params.recurrent_dropout_prob

  rnn_cell = get_rnn_cell(
    enc_cell_fn, params.enc_rnn_size,
    use_dropout=is_training and params.use_recurrent_dropout,
    keep_prob=recurrent_dropout_prob, is_bidir=params.use_bidirection_lstm)
  if params.use_bidirection_lstm:
    enc_cell_fw, enc_cell_bw = rnn_cell[0], rnn_cell[1]
  else:
    enc_cell_fw = rnn_cell
    enc_cell_bw = None
  #########################
  ## Network Declaration ##
  #########################
  seq_encoder = _get_network(ENC_FN)

  outputs = dict()
  ##############################
  ## Encoding T = f(b) - f(a) ##
  ##############################
  with tf.variable_scope('seq_enc', reuse=reuse):
    tmp_outputs = seq_encoder(
      inputs['A_landmarks'], inputs['A_lens'],
      enc_cell_fw, enc_cell_bw, params, is_training=is_training)
  outputs['A_features'] = tmp_outputs['features']
  outputs['A_content'] = tmp_outputs['content']
  outputs['A_style'] = tmp_outputs['style']

  with tf.variable_scope('seq_enc', reuse=True):
    tmp_outputs = seq_encoder(
      inputs['B_landmarks'], inputs['B_lens'],
      enc_cell_fw, enc_cell_bw, params, is_training=is_training)
  outputs['B_features'] = tmp_outputs['features']
  outputs['B_content'] = tmp_outputs['content']
  outputs['B_style'] = tmp_outputs['style']

  with tf.variable_scope('seq_enc', reuse=True):
   tmp_outputs = seq_encoder(
      inputs['C_landmarks'], inputs['C_lens'],
      enc_cell_fw, enc_cell_bw, params, is_training=is_training)
  outputs['C_features'] = tmp_outputs['features']
  outputs['C_content'] = tmp_outputs['content']
  outputs['C_style'] = tmp_outputs['style']
  return outputs

def analogy_seq_decoding_model(inputs, outputs, params, is_training, reuse, output_length=None):
  assert (not is_training)
  dec_cell_fn = NAME_TO_RNNCELL[params.dec_model]
  
  recurrent_dropout_prob = 1.0
  if is_training:
    recurrent_dropout_prob = params.recurrent_dropout_prob

  dec_cell = get_rnn_cell(
    dec_cell_fn, params.dec_rnn_size,
    use_dropout=is_training and params.use_recurrent_dropout,
    keep_prob=recurrent_dropout_prob, is_bidir=False)

  fut_decoder = _get_network(FUT_DEC_FN)
  ##############
  ## Decoding ##
  ##############
  if output_length is None:
    output_length = params.max_length

  prev_state = outputs['dec_D_embedding']
  if hasattr(params, 'dec_style') and params.dec_style > 0:
    dec_style = outputs['D_style']
  else:
    dec_style = None
  with tf.variable_scope('fut_dec', reuse=reuse):
    tmp_outputs = fut_decoder(
      prev_state, dec_style, dec_cell, output_length, params, is_training=is_training)
  outputs['D_landmarks'] = tmp_outputs['keypoint_output']
  return outputs

