"""Human3.6M keypoint Sequence Encoder/Decoder factory."""

import tensorflow as tf

from factory_utils import get_network, get_rnn_cell
import h36m_seq_encoder as keypoint_seq_encoder
import h36m_seq_decoder as keypoint_seq_decoder
import rnn
import numpy as np
import random

ENC_FN = 'keypoint_seq_encoder'
FUT_DEC_FN = 'keypoint_seq_decoder'

NAME_TO_NETS = {
  ENC_FN: keypoint_seq_encoder,
  FUT_DEC_FN: keypoint_seq_decoder,
} 

NAME_TO_RNNCELL = {
  'lstm': rnn.LSTMCell,
  'ln_lstm': rnn.LayerNormLSTMCell,
}

def _get_network(name):
  return get_network(name, NAME_TO_NETS)

def get_seq_encoding_model(inputs, params, is_training, reuse):
  """Factory function to retrieve encoder network model."""
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
  seq_encoder = _get_network(ENC_FN)
 
  outputs = dict()
  #######################
  ## Encoding function ##
  #######################
  with tf.variable_scope('seq_enc', reuse=reuse):
    tmp_outputs = seq_encoder(
      inputs['his_landmarks'], inputs['his_lens'],
      enc_cell_fw, enc_cell_bw, params, is_training=is_training)
  outputs['his_features'] = tmp_outputs['features']
  if hasattr(params, 'content_dim'):
    outputs['his_content'] = tmp_outputs['content']
    outputs['his_style'] = tmp_outputs['style']

  with tf.variable_scope('seq_enc', reuse=True):
    tmp_outputs = seq_encoder(
      inputs['fut_landmarks'], inputs['fut_lens'],
      enc_cell_fw, enc_cell_bw, params, is_training=is_training)
  outputs['fut_features'] = tmp_outputs['features']
  if hasattr(params, 'content_dim'):
    outputs['fut_content'] = tmp_outputs['content']
    outputs['fut_style'] = tmp_outputs['style']
  return outputs

def get_seq_decoding_model(inputs, outputs, params, is_training, reuse, output_length=None):
  dec_cell_fn = NAME_TO_RNNCELL[params.dec_model]
  
  recurrent_dropout_prob = 1.0
  if is_training:
    recurrent_dropout_prob = params.recurrent_dropout_prob

  dec_cell = get_rnn_cell(
    dec_cell_fn, params.dec_rnn_size,
    use_dropout=is_training and params.use_recurrent_dropout,
    keep_prob=recurrent_dropout_prob, is_bidir=False)

  fut_decoder = _get_network(FUT_DEC_FN)
  if output_length is None:
    output_length = params.max_length
   
  prev_state = outputs['dec_embedding']
  if hasattr(params, 'dec_style') and params.dec_style > 0:
    dec_style = outputs['dec_style']
  else:
    dec_style = None
  #
  with tf.variable_scope('fut_dec', reuse=reuse):
    tmp_outputs = fut_decoder(
      prev_state, dec_style, dec_cell, output_length, params, is_training=is_training)
  outputs['fut_landmarks'] = tmp_outputs['keypoint_output']
  return outputs

def get_cycle_decoding_model(inputs, outputs, params, is_training, reuse):
  assert is_training
  dec_cell_fn = NAME_TO_RNNCELL[params.dec_model]
  
  recurrent_dropout_prob = 1.0
  if is_training:
    recurrent_dropout_prob = params.recurrent_dropout_prob
  
  dec_cell = get_rnn_cell(
    dec_cell_fn, params.dec_rnn_size,
    use_dropout=is_training and params.use_recurrent_dropout,
    keep_prob=recurrent_dropout_prob, is_bidir=False)

  fut_decoder = _get_network(FUT_DEC_FN)
  ############################
  ## Generate Future Frames ##
  ############################
  prev_state = outputs['cycle_dec_embedding']
  if hasattr(params, 'dec_style') and params.dec_style > 0:
    cycle_style = outputs['cycle_style']
  else:
    cycle_style = None
  with tf.variable_scope('fut_dec', reuse=True):
    tmp_outputs = fut_decoder(
      prev_state, cycle_style, dec_cell, params.max_length, params, is_training)
  outputs['cycle_fut_landmarks'] = tmp_outputs['keypoint_output']
  return outputs

