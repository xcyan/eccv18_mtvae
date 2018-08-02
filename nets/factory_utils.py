"""Helper functions for Motion Generation."""

import tensorflow as tf

def get_network(name, name_to_nets):
  """Gets a single network component."""
  if name not in name_to_nets:
    raise ValueError('Network name [%s] not recognized.' % name)
  return name_to_nets[name].model

def get_rnn_cell(cell_fn, rnn_size, use_dropout, keep_prob, is_bidir=False):
  rnn_cell_fw = cell_fn(
    rnn_size,
    use_recurrent_dropout=use_dropout,
    dropout_keep_prob=keep_prob)
  if not is_bidir:
    return rnn_cell_fw
  rnn_cell_bw = cell_fn(
    rnn_size,
    use_recurrent_dropout=use_dropout,
    dropout_keep_prob=keep_prob)
  return rnn_cell_fw, rnn_cell_bw


