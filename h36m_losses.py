"""Define several loss function for KeypointVAE."""

import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

EPS = 1e-9

def safe_div(nom_term, denom_term):
  return tf.div(nom_term, denom_term+EPS)

def safe_log(num_term):
  return tf.log(num_term+EPS)

def safe_sqrt(num_term):
  return tf.sqrt(num_term+EPS)

def tf_gaussian_kl(post_mu, post_logs2, prior_mu=0.0, prior_logs2=0.0):
  kl_matrix = 0.5 * (-1.0 + prior_logs2 - post_logs2)
  kl_matrix += 0.5 * safe_div(
    tf.exp(post_logs2) + tf.square(prior_mu-post_mu),
    tf.exp(prior_logs2))
  kl = tf.reduce_sum(kl_matrix)
  return kl

def get_keypoint_l2_matrix(inputs, outputs, max_length):
  shp = inputs['fut_landmarks'].get_shape().as_list()
  quantity, keypoint_dim = shp[0], shp[2]
  fut_landmarks = tf.reshape(
    inputs['fut_landmarks'], [quantity, max_length, keypoint_dim * 2])
  pred_landmarks = tf.reshape(
    outputs['fut_landmarks'], [quantity, max_length, keypoint_dim * 2])
  l2_matrix = tf.reduce_sum(tf.square(fut_landmarks-pred_landmarks), 2)
  return l2_matrix

def get_keypoint_loss(inputs, outputs,
                                max_length, weight_scale):
  shp = inputs['fut_landmarks'].get_shape().as_list()
  quantity, keypoint_dim = shp[0], shp[2]
  #num_mixture = outputs['fut_mu'].get_shape().as_list()[2]
  fut_landmarks = tf.reshape(
    inputs['fut_landmarks'], [quantity, max_length, keypoint_dim * 2])
  pred_landmarks = tf.reshape(
    outputs['fut_landmarks'], [quantity, max_length, keypoint_dim * 2])
  seq_mask = tf.sequence_mask(
    inputs['fut_lens'], maxlen=max_length,
    dtype=tf.float32)
  seq_mask = tf.reshape(seq_mask, [quantity, max_length, 1])
  result = tf.reduce_sum(tf.abs(
    seq_mask * fut_landmarks - seq_mask * pred_landmarks))
  keypoint_loss = tf.reduce_sum(result)
  keypoint_loss /= tf.to_float(quantity * keypoint_dim)
  slim.summaries.add_scalar_summary(
    keypoint_loss, 'posterior_keypoint_loss', prefix='losses')
  keypoint_loss *= weight_scale
  return keypoint_loss

def get_velocity_loss(last_features, fut_features, pred_features,
                      fut_lens, weight_scale, name, velocity_length=3):
  shp = fut_features.get_shape().as_list()
  quantity, max_length, feature_dim = shp[0], shp[1], shp[2]
  gt_velocity = fut_features - tf.concat([last_features, fut_features[:, :-1]], axis=1)
  pred_velocity = pred_features - tf.concat([last_features, pred_features[:, :-1]], axis=1)
  #
  fut_lens = tf.minimum(fut_lens, velocity_length)
  #
  seq_mask = tf.sequence_mask(
    fut_lens, maxlen=max_length, dtype=tf.float32)
  seq_mask = tf.reshape(seq_mask, [quantity, max_length, 1, 1])
  result = tf.reduce_sum(
    tf.abs(seq_mask * gt_velocity - seq_mask * pred_velocity))
  velocity_loss = result / tf.to_float(quantity * feature_dim)
  slim.summaries.add_scalar_summary(
    velocity_loss, name, prefix='losses')
  velocity_loss *= weight_scale
  return velocity_loss

def get_kl_loss(inputs, outputs, weight_scale, kl_tolerance=0.2):
  quantity = inputs['fut_landmarks'].get_shape().as_list()[0]
  noise_dim = outputs['mu_latent'].get_shape().as_list()[1]

  kl_loss = tf_gaussian_kl(
    outputs['mu_latent'], outputs['logs2_latent'])
  kl_loss /= tf.to_float(quantity * noise_dim)
  slim.summaries.add_scalar_summary(
    kl_loss, 'kl_loss', prefix='losses')
  kl_loss = tf.maximum(kl_tolerance, kl_loss)
  kl_loss *= weight_scale
  return kl_loss

def get_cycle_loss(inputs, outputs, weight_scale):
  shp = outputs['cycle_latent'].get_shape().as_list()
  quantity, noise_dim = shp[0], shp[1]
  #
  y = outputs['cycle_latent']
  mu = outputs['cycle_mu_latent']
  logs2 = outputs['cycle_logs2_latent']
  s = tf.exp(0.5 * logs2)
  y2 = mu + s * tf.random_normal([quantity, noise_dim])
  #
  cycle_loss = tf.reduce_sum(tf.abs(y-y2))
  cycle_loss /= tf.to_float(quantity * noise_dim)
  slim.summaries.add_histogram_summary(
    y, name='cycle_latent', prefix='summaries')
  slim.summaries.add_histogram_summary(
    mu, name='cycle_mu_latent', prefix='summaries')
  slim.summaries.add_scalar_summary(
    cycle_loss, 'cycle_loss', prefix='losses')
  cycle_loss *= weight_scale
  return cycle_loss


def regularization_loss(scopes, params):
  """Computes the weight decay as regularization during training."""
  reg_loss = tf.zeros(dtype=tf.float32, shape=[])
  if params.weight_decay > 0:
    is_trainable = lambda x: x in tf.trainable_variables()
    # TODO(xcyan): double check this.
    is_weights = lambda x: 'weights' in x.name
    for scope in scopes:
      scope_vars = filter(is_trainable,
                          tf.contrib.framework.get_model_variables(scope))
      scope_vars = filter(is_weights, scope_vars)
      if scope_vars:
        reg_loss += tf.add_n([tf.nn.l2_loss(var) for var in scope_vars])
  
  slim.summaries.add_scalar_summary(
    reg_loss, 'reg_loss', prefix='losses')
  reg_loss *= params.weight_decay
  return reg_loss
 

