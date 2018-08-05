"""Base class for Human3.6M Keypoint Generation."""

import os
import numpy as np
import tensorflow as tf

import h36m_input as input_generator
import h36m_losses as losses
import utils
import model_utils
from preprocess.video_proc_utils import VideoProc

slim = tf.contrib.slim


def _get_data_from_provider(inputs, batch_size, 
                            split_name, is_training=True, load_image=False):
  """Load data from input_genetator."""
  input_tuple = [inputs['landmarks']]
  if load_image:
    input_tuple.append(inputs['images'])
  
  tmp_outputs = tf.train.batch(input_tuple,
    batch_size=batch_size,
    num_threads=64,
    capacity=batch_size*4,
    name='batching_queues/%s' % split_name)
  
  outputs = dict()
  outputs['dataset_size'] = inputs['dataset_size']
  
  if load_image:
    outputs['landmarks'] = tmp_outputs[0]
    outputs['images'] = tmp_outputs[1]
  else:
    outputs['landmarks'] = tmp_outputs

  return outputs

  
class H36M_BasePredModel(object):
  """Defines Human3.6M motion generation model."""

  def __init__(self, params):
    self._params = params

  def get_inputs_from_placeholder(self, dataset_size, batch_size): 
    params = self._params
    init_length = params.max_input_length
    assert params.max_input_length == params.min_input_length
    sample_length = init_length + params.max_length
    #
    placeholder = dict()
    placeholder['landmarks'] = tf.placeholder(
      dtype=tf.float32, shape=(batch_size, sample_length, params.keypoint_dim, 2))
    #
    inputs = dict()
    inputs.update(placeholder)
    inputs['dataset_size'] = dataset_size
    inputs['his_lens'] = tf.tile([init_length], [batch_size])
    inputs['fut_lens'] = tf.tile([params.max_length], [batch_size])
    inputs['parzen_radius'] = tf.placeholder(
      dtype=tf.float32, shape=())
    return inputs

  def get_duplicate_inputs(self, dataset_dir, dataset_name, split_name,
                           batch_size, load_image):
    """Loads a batch of input from a single source."""    
    params = self._params
    init_length = params.max_input_length
    sample_length = init_length + params.max_length
    assert params.max_input_length == params.min_input_length
    with tf.variable_scope('data_loading_%s/%s' % (dataset_name, split_name)):
      raw_inputs = input_generator.get_tfrecord(
        dataset_dir, dataset_name, split_name,
        shuffle=False, sample_length=sample_length)
      inputs = _get_data_from_provider(
        raw_inputs, batch_size, split_name,
        is_training=True, load_image=load_image)
      inputs['landmarks'] = tf.tile(
        inputs['landmarks'][0:1], [batch_size, 1, 1, 1])
      inputs['his_lens'] = tf.tile([init_length], [batch_size])
      inputs['fut_lens'] = tf.tile([params.max_length], [batch_size])
      assert (not load_image)
    return inputs

  def get_inputs(self, dataset_dir, dataset_name, split_name,
                 batch_size, is_training):
    """Loads given dataset and split."""
    params = self._params
    sample_length = params.max_input_length + params.max_length
    with tf.variable_scope('data_loading_%s/%s' % (dataset_name, split_name)):
      raw_inputs = input_generator.get_tfrecord(
        dataset_dir, dataset_name, split_name, 
        shuffle=is_training, sample_length=sample_length)
      inputs = _get_data_from_provider(
        raw_inputs, batch_size, split_name, is_training)
      #
      if params.min_input_length < params.max_input_length:
        inputs['his_lens'] = tf.random_uniform(
          [batch_size], minval=params.min_input_length, maxval=params.max_input_length,
           dtype=tf.int32)
      else:
        inputs['his_lens'] = tf.tile([params.max_input_length], [batch_size])
      inputs['fut_lens'] = tf.tile([params.max_length], [batch_size])
    return inputs

  def preprocess(self, raw_inputs, is_training, load_image=False):
    """Data augmentation."""
    params = self._params
    shp = raw_inputs['landmarks'].get_shape().as_list()
    quantity, max_length = shp[0], shp[1]
    inputs = dict()
    inputs['dataset_size'] = raw_inputs['dataset_size']
    inputs['his_lens'] = raw_inputs['his_lens']
    inputs['fut_lens'] = raw_inputs['fut_lens']
    inputs['his_landmarks'], inputs['fut_landmarks'] = \
      input_generator.split_pts_seq(
        raw_inputs['landmarks'], inputs['his_lens'], inputs['fut_lens'],
        params.max_input_length, params.max_length, is_training)
    seq_idx = tf.range(0, quantity, dtype=tf.int32, name='range')
    seq_idx = tf.stack([seq_idx, inputs['his_lens']], axis=1)
    inputs['last_landmarks'] = tf.gather_nd(inputs['his_landmarks'], seq_idx)
    inputs['last_landmarks'] = tf.reshape(
      inputs['last_landmarks'], [quantity, 1, params.keypoint_dim, 2])
    assert (not load_image)
    return inputs

  def save_html_page(self, save_dir, num_repeat, batch_size):
    content = '<html><body><h1>Visualization</h1><table boder="1" style="width=100%">'
    content = content + '<tr><td>ground truth</td><td>pred (multiple trajectories)</td>'
    content = content + '</td>'
    for t in xrange(num_repeat):
      content = content + '\n<tr>'
      content = content + '<td><img src=\"%02d_gt.gif\" style=\"width:255px;height:255px;\"></td>' % t 
    
      content += '<td><img src=\"%02d_pred.gif\" style=\"width:%dpx;height:255px;\"></td>' % (t, batch_size * 255)
      content += '</tr>'
    content += '</html>'
    with open(os.path.join(save_dir, 'index.html'), 'w') as f:
      f.write(content)
      f.flush()
    f.close()

  def get_init_fn(self, scopes):
    """Initialize assignment operator function used while training."""
    return model_utils.get_init_fn(scopes, self._params.init_model)
 
  def get_train_op_for_scope(self, loss, optimizer, scopes):
    train_op = model_utils.get_train_op_for_scope(
      loss, optimizer, scopes, self._params.clip_gradient_norm)
    return train_op

  def get_regularization_loss(self, outputs, scopes):
    params = self._params
    reg_loss = losses.regularization_loss(scopes, params)
    return reg_loss
   
  def get_visualization_op(self, inputs, outputs, log_dir, counter, output_length=None):
    params = self._params
    batch_size = self._params.batch_size
    max_length = self._params.max_length
    img_size = self._params.img_size

    ob_lens = inputs['his_lens']
    gt_lens = inputs['fut_lens']
    if output_length is None:
      pred_lens = gt_lens
    else:
      pred_lens = tf.zeros_like(gt_lens, dtype=tf.int32) + output_length
    ob_landmarks = inputs['his_landmarks'] * params.img_size
    pred_landmarks = outputs['fut_landmarks'] * params.img_size
    gt_landmarks = inputs['fut_landmarks'] * params.img_size    
    
    def write_grid(base_id, his_lens, gt_lens, pred_lens,
                   his_landmarks, gt_landmarks, pred_landmarks):
      """Python function."""
      tmp_dir = os.path.join(log_dir, 'tmp')
      utils.force_mkdir(tmp_dir)
      video_proc_lib = VideoProc()
      #############################
      ## Plot the history frames ##
      #############################
      his_video = np.zeros((his_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(his_lens[0]):
        his_video[t] = utils.visualize_landmarks(his_video[t], his_landmarks[0][t])
        his_video[t] = utils.visualize_h36m_skeleton(his_video[t], his_landmarks[0][t])
        his_video[t] = utils.visualize_boundary(his_video[t], colormap='green')
      #################################
      ## Plot the gt (future) frames ##
      #################################
      gt_video = np.zeros((gt_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(gt_lens[0]):
        gt_video[t] = utils.visualize_landmarks(gt_video[t], gt_landmarks[0][t])
        gt_video[t] = utils.visualize_h36m_skeleton(gt_video[t], gt_landmarks[0][t])
        gt_video[t] = utils.visualize_boundary(gt_video[t], colormap='blue')
      merged_video = np.concatenate((his_video, gt_video), axis=0)
      video_proc_lib.save_img_seq_to_video(
        merged_video, log_dir, '%02d_gt.gif' % base_id, frame_rate=7.5, codec=None, override=True)
      ###################################
      ## Plot the pred (future) frames ##
      ###################################
      raw_gif_list = []
      for i in xrange(batch_size):
        print(base_id * batch_size + i) 
        pred_video = np.zeros((pred_lens[i], img_size, img_size, 3), dtype=np.float32)
        for t in xrange(pred_lens[i]):
          pred_video[t] = utils.visualize_landmarks(pred_video[t], pred_landmarks[i][t])
          pred_video[t] = utils.visualize_h36m_skeleton(pred_video[t], pred_landmarks[i][t])
          pred_video[t] = utils.visualize_boundary(pred_video[t], colormap='red')
        merged_video = np.concatenate((his_video, pred_video), axis=0)
        video_proc_lib.save_img_seq_to_video(
          merged_video, log_dir, '%02d_pred%02d.gif' % (base_id, i),
          frame_rate=7.5, codec=None, override=True)
        raw_gif_list.append('%02d_pred%02d.gif' % (base_id, i))
      video_proc_lib.merge_video_side_by_side(
        log_dir, raw_gif_list, '%02d_pred.gif' % base_id,
        override=True)
        
      return 0
    save_op = tf.py_func(
      func=write_grid,
      inp=[counter, ob_lens, gt_lens, pred_lens, ob_landmarks, gt_landmarks, pred_landmarks],
      Tout=[tf.int64], name='write_grid')[0]
    return save_op

  

