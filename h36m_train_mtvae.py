"""Contains training plan for sequence generation using MTVAE on Human3.6M."""

import os
import numpy as np
import tensorflow as tf

from tensorflow import app
from H36M_MTVAEPredModel import H36M_MTVAEPredModel
import utils
import copy

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir', '', '')
flags.DEFINE_string('dataset_name', None, '')
flags.DEFINE_integer('keypoint_dim', 32, '')
flags.DEFINE_integer('batch_size', 256, '')
flags.DEFINE_integer('min_input_length', 10, '')
flags.DEFINE_integer('max_input_length', 20, '')
flags.DEFINE_integer('max_length', 64, '')
#
flags.DEFINE_integer('skip_n_frame', 1, '')
flags.DEFINE_integer('img_size', 180, '')
flags.DEFINE_integer('noise_dim', 512, '')
flags.DEFINE_integer('content_dim', 512, '')
flags.DEFINE_integer('embed_dim', 1024, '')
flags.DEFINE_integer('enc_rnn_size', 1024, '')
flags.DEFINE_integer('dec_rnn_size', 1024, '')
flags.DEFINE_integer('enc_fc_layers', 0, '')
flags.DEFINE_integer('latent_fc_layers', 1, '')
flags.DEFINE_integer('dec_fc_layers', 1, '')
flags.DEFINE_integer('use_latent', 1, '')
flags.DEFINE_integer('use_prior', 0, '')
flags.DEFINE_string('enc_model', 'ln_lstm', '')
flags.DEFINE_string('dec_model', 'ln_lstm', '')
flags.DEFINE_integer('use_bidirection_lstm', 0, '')
flags.DEFINE_string('dec_interaction', 'add', '')
flags.DEFINE_integer('T_layer_norm', 1, '')
flags.DEFINE_integer('dec_style', 1, '')
# Save options.
flags.DEFINE_string('checkpoint_dir', None, '')
flags.DEFINE_string('model_name', None, '')
flags.DEFINE_string('init_model', None, '')
flags.DEFINE_integer('save_every', 500, '')
# Optimization.
flags.DEFINE_integer('use_recurrent_dropout', 0, '')
flags.DEFINE_float('recurrent_dropout_prob', 1.0, '')
#
flags.DEFINE_float('keypoint_weight', 1, '')
flags.DEFINE_integer('velocity_length', 8, '')
flags.DEFINE_float('velocity_weight', 5.0, '')
flags.DEFINE_float('velocity_start_weight', 1e-5, '')
flags.DEFINE_float('velocity_decay_rate', 0.99995, '')
flags.DEFINE_float('kl_start_weight', 1e-5, '')
flags.DEFINE_float('kl_weight', 1.0, '')
flags.DEFINE_float('kl_decay_rate', 0.99995, '')
flags.DEFINE_float('kl_tolerance', 0.01, '')
flags.DEFINE_integer('cycle_model', 1, '')
flags.DEFINE_float('cycle_weight', 5.0, '')
#
flags.DEFINE_float('learning_rate', 0.0001, '')
flags.DEFINE_float('weight_decay', 1e-12, '')
flags.DEFINE_float('clip_gradient_norm', 0, '')
flags.DEFINE_integer('max_number_of_steps', 5000, '')
# Summary.
flags.DEFINE_integer('save_summaries_secs', 60, '')
flags.DEFINE_integer('save_interval_secs', 60 * 5, '')
# Scheduling.
flags.DEFINE_string('master', '', '')
flags.DEFINE_bool('sync_replicas', False, '')
flags.DEFINE_integer('worker_replicas', 1, '')
flags.DEFINE_integer('backup_workers', 0, '')
flags.DEFINE_integer('ps_tasks', 0, '')
flags.DEFINE_integer('task', 0, '')

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
      model = H36M_MTVAEPredModel(FLAGS)
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

