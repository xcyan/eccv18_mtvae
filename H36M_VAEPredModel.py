"""Helper functions for training Vanilla VAE on Human3.6M."""

import os
import numpy as np
import tensorflow as tf

import h36m_losses as losses
import H36M_BasePredModel as BasePredModel
from nets import h36m_vae_factory as model_factory

slim = tf.contrib.slim

class VAEPredModel(BasePredModel.BasePredModel):
  """Defines VAE Prediciton Model."""

  def __init__(self, params):
    super(VAEPredModel, self).__init__(params)

  def get_model_fn(self, is_training, use_prior, reuse):
    params = self._params
    model_fn = model_factory.get_model_fn(self._params, is_training,
                                            use_prior, reuse)
    return model_fn

  def get_sample_fn(self, is_training, use_prior, reuse, output_length=None):
    return model_factory.get_sample_fn(self._params, is_training,
                                       use_prior, reuse, output_length)

  def get_loss(self, step, inputs, outputs):
    total_loss = tf.zeros(dtype=tf.float32, shape=[])
    loss_dict = dict()
    params = self._params

    if hasattr(params, 'keypoint_weight') and (params.keypoint_weight > 0):
      keypoint_loss = losses.get_keypoint_loss(
        inputs, outputs, params.max_length, params.keypoint_weight)
      loss_dict['post_keypoint_loss'] = keypoint_loss
      total_loss += keypoint_loss
    
    if hasattr(params, 'velocity_weight') and (params.velocity_weight > 0):
      curr_velocity_weight = (params.velocity_weight - (params.velocity_weight - params.velocity_start_weight) * (params.velocity_decay_rate)**tf.to_float(step))
      velocity_loss = losses.get_velocity_loss(
        inputs['last_landmarks'], inputs['fut_landmarks'], outputs['fut_landmarks'],
        inputs['fut_lens'], curr_velocity_weight * params.keypoint_weight,
        'post_velocity_loss', params.velocity_length)
      loss_dict['velocity_loss'] = velocity_loss
      total_loss += velocity_loss

    if hasattr(params, 'kl_weight') and (params.kl_weight > 0):
      curr_kl_weight = (params.kl_weight - (params.kl_weight - params.kl_start_weight) * 
                       (params.kl_decay_rate)**tf.to_float(step))     
      kl_loss = losses.get_kl_loss(
        inputs, outputs, curr_kl_weight, params.kl_tolerance)
      loss_dict['kl_loss'] = kl_loss
      total_loss += kl_loss

    slim.summaries.add_scalar_summary(
      total_loss, 'keypoint_vae_loss', prefix='losses')
    return total_loss, loss_dict

  def print_running_loss(self, global_step, loss_dict):
    params = self._params
    if params.keypoint_weight > 0:
      norm_keypoint_loss = loss_dict['post_keypoint_loss'] / params.keypoint_weight
    else:
      norm_keypoint_loss = 0

    if params.kl_weight > 0:
      curr_kl_weight = (params.kl_weight - (params.kl_weight - params.kl_start_weight) * 
                       (params.kl_decay_rate)**tf.to_float(global_step))
      norm_kl_loss = loss_dict['kl_loss'] / curr_kl_weight
    else:
      norm_kl_loss = 0
    
    if hasattr(params, 'velocity_weight') and params.velocity_weight > 0:
      curr_velocity_weight = (params.velocity_weight - (params.velocity_weight - params.velocity_start_weight) * (params.velocity_decay_rate)**tf.to_float(global_step))
      norm_velocity_loss = loss_dict['velocity_loss'] / (curr_velocity_weight * params.keypoint_weight)
    else:
      norm_velocity_loss = 0

    def print_loss(step, keypoint_loss, kl_loss, velocity_loss):
      print('[%06d]\t[Keypoint %.3f]\t[KL %.3f]\t[VF %.3f]' % \
        (step, keypoint_loss, kl_loss, velocity_loss)) 
      return 0
    ret_tmp = tf.py_func(
      func=print_loss,
      inp=[global_step, norm_keypoint_loss, norm_kl_loss, 
            norm_velocity_loss],
      Tout=[tf.int64], name='print_loss')[0]
    ret_tmp = tf.to_int32(ret_tmp)
    return ret_tmp
