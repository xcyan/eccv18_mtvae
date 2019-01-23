"""Contains training plan for sequence generation using MTVAE on Human3.6M."""

import os
import numpy as np
import tensorflow as tf

from tensorflow import app
from H36M_MTVAEPredModel import MTVAEPredModel
import utils
import copy

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir', 'workspace/', 'Directory path containing the input data.')
flags.DEFINE_string('dataset_name', None, 'Name of the dataset that is to be used for training or evaluation.')
flags.DEFINE_integer('keypoint_dim', 32, 'Number of the human joints.')
flags.DEFINE_integer('batch_size', 256, 'Batch size for training.')
flags.DEFINE_integer('min_input_length', 10, 'Minimum number of frames loaded as initial motion.')
flags.DEFINE_integer('max_input_length', 20, 'Maximum number of frames loaded as initial motion.')
flags.DEFINE_integer('max_length', 64, 'Maximum number of frames to predict as future motion.')
# Model parameters.
flags.DEFINE_integer('skip_n_frame', 1, 'Subsample rate for constructing the training data.')
flags.DEFINE_integer('img_size', 180, 'Input frame dimension (pixels) for visualization purpose.')
flags.DEFINE_integer('noise_dim', 512, 'Latent variable dimension.')
flags.DEFINE_integer('content_dim', 512, 'Deterministic stream dimension.')
flags.DEFINE_integer('embed_dim', 1024, 'Dimension used as input/output of LSTM layer.')
flags.DEFINE_integer('enc_rnn_size', 1024, 'Encoder LSTM dimension.')
flags.DEFINE_integer('dec_rnn_size', 1024, 'Decoder LSTM dimension.')
flags.DEFINE_integer('enc_fc_layers', 0, 'Number of fully-connected layer(s) applied to human joints.')
flags.DEFINE_integer('latent_fc_layers', 1, 'Number of fully-connected layer(s) applied to latent variable.')
flags.DEFINE_integer('dec_fc_layers', 1, 'Number of fully-connected layer(s) applied in decoding module.')
flags.DEFINE_integer('use_latent', 1, 'If True, use (stochastic) latent variable for motion prediction.')
flags.DEFINE_integer('use_prior', 0, 'If True, sample latent variable from prior distribution.')
flags.DEFINE_string('enc_model', 'ln_lstm', 'Encoder LSTM module.')
flags.DEFINE_string('dec_model', 'ln_lstm', 'Decoder LSTM module.')
flags.DEFINE_integer('use_bidirection_lstm', 0, 'If True, use bi-directional LSTM module.')
flags.DEFINE_string('dec_interaction', 'add', 'The interaction mechanism used for latent and deterministic stream.')
flags.DEFINE_integer('T_layer_norm', 1, 'If True, apply layer normalization on the intermediate fully-connected layers.')
flags.DEFINE_integer('dec_style', 1, 'If True, add connection to every step of decoder LSTM.')
# Save options.
flags.DEFINE_string('checkpoint_dir', None, 'Directory path for saving trained models and other data.')
flags.DEFINE_string('model_name', None, 'Name of the model used in naming the TF job. Must be different for each run.')
flags.DEFINE_string('init_model', None, 'Checkpoint path of the model to initialize with.')
flags.DEFINE_integer('save_every', 500, 'Average period of steps after which we save a model.')
# Optimization.
flags.DEFINE_integer('use_recurrent_dropout', 0, 'If True, apply dropout to recurrent layer(s).')
flags.DEFINE_float('recurrent_dropout_prob', 1.0, 'The dropout ratio.')
# Loss.
flags.DEFINE_float('keypoint_weight', 1, 'Weighting factor for (reconstructed) human joints.')
flags.DEFINE_integer('velocity_length', 8, 'Number of steps to be considered while applying velocity constraint.')
flags.DEFINE_float('velocity_weight', 5.0, 'Weighting factor for the velocity of (reconstructed) human joints.')
flags.DEFINE_float('velocity_start_weight', 1e-5, 'The initial annealing weight for velocity constraint.')
flags.DEFINE_float('velocity_decay_rate', 0.99995, 'Annealing decay rate for velocity constraint.')
flags.DEFINE_float('kl_start_weight', 1e-5, 'The initial annealing weight for KL divergence.')
flags.DEFINE_float('kl_weight', 1.0, 'Weighting factor for KL divergence.')
flags.DEFINE_float('kl_decay_rate', 0.99995, 'Annealing decay rate for KL divergence.')
flags.DEFINE_float('kl_tolerance', 0.01, 'Level of KL loss at which to stop optimizing for KL.')
flags.DEFINE_integer('cycle_model', 1, 'If True, apply cycle constraint while training the model.')
flags.DEFINE_float('cycle_weight', 5.0, 'Weighting factor for cycle constraint.')
# Learning steps.
flags.DEFINE_float('learning_rate', 0.0001, 'Learning rate.')
flags.DEFINE_float('weight_decay', 1e-12, 'Weight decay parameter while training.')
flags.DEFINE_float('clip_gradient_norm', 0, 'Gradient clip norm, leave 0 if no gradient clipping.')
flags.DEFINE_integer('max_number_of_steps', 5000, 'Maximum number of steps for training.')
# Summary.
flags.DEFINE_integer('save_summaries_secs', 60, 'Seconds interval for dumping TF summaries.')
flags.DEFINE_integer('save_interval_secs', 60 * 5, 'Seconds interval to save models.')
# Scheduling.
flags.DEFINE_string('master', '', 'The address of the tensorflow master.')
flags.DEFINE_bool('sync_replicas', False, 'Whether to sync gradients between replicas for optimizer.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker replicas (train tasks).')
flags.DEFINE_integer('backup_workers', 0, 'Number of backup workers.')
flags.DEFINE_integer('ps_tasks', 0, 'Number of ps tasks.')
flags.DEFINE_integer('task', 0, 'Task identifier flag to be set for each task running in distributed manner. Task number 0 '
                     'will be chosen as the chief.')

FLAGS = flags.FLAGS

def main(_):
  train_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name, 'train')
  utils.force_mkdir(os.path.join(FLAGS.checkpoint_dir, FLAGS.model_name))
  utils.force_mkdir(train_dir)

  g = tf.Graph()
  with g.as_default():
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      global_step = slim.get_or_create_global_step()
      ###########
      ## model ##
      ###########
      model = MTVAEPredModel(FLAGS)
      ##########
      ## data ##
      ##########
      train_data = model.get_inputs(
        FLAGS.inp_dir, FLAGS.dataset_name, 'train',
        FLAGS.batch_size, is_training=True)
      inputs = model.preprocess(train_data, is_training=True)
      ##############
      ## model_fn ##
      ##############
      model_fn = model.get_model_fn(
        is_training=True, use_prior=FLAGS.use_prior, reuse=False)
      outputs = model_fn(inputs)
      ##################
      ## train_scopes ##
      ##################
      train_scopes = ['seq_enc', 'latent_enc', 'latent_dec', 'fut_dec']
      init_scopes = train_scopes
      if FLAGS.init_model:
        init_fn = model.get_init_fn(init_scopes)
      else:
        init_fn = None
      ##########
      ## loss ##
      ##########
      total_loss, loss_dict = model.get_loss(global_step, inputs, outputs)
      reg_loss = model.get_regularization_loss(outputs, train_scopes)
      print_op = model.print_running_loss(global_step, loss_dict) 
      ###############
      ## optimizer ##
      ###############
      optimizer = tf.train.AdamOptimizer(
        FLAGS.learning_rate, beta1=0.9, beta2=0.999)
       
      ##############
      ## train_op ##
      ##############
      train_op = model.get_train_op_for_scope(
        total_loss+reg_loss,
        optimizer, train_scopes)
      with tf.control_dependencies([print_op]):
        train_op = tf.identity(train_op)
      ###########
      ## saver ##
      ###########
      saver = tf.train.Saver(max_to_keep=np.minimum(5, FLAGS.worker_replicas+1))
      ##############
      ## training ##
      ##############      
      slim.learning.train(
        train_op=train_op,
        logdir=train_dir,
        init_fn=init_fn,
        master=FLAGS.master,
        is_chief=(FLAGS.task==0),
        number_of_steps=FLAGS.max_number_of_steps,
        saver=saver,
        save_summaries_secs=FLAGS.save_summaries_secs,
        save_interval_secs=FLAGS.save_interval_secs)

if __name__ == '__main__':
  app.run()

