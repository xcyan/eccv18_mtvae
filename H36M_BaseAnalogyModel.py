"""Helper Base Class functions for Human3.6M keypoint analogy-making."""

import os
import numpy as np
import tensorflow as tf

import h36m_input as input_generator
from H36M_BasePredModel import _get_data_from_provider, BasePredModel
import utils
from preprocess.video_proc_utils import VideoProc

class BaseAnalogyModel(BasePredModel):
  """Defines analogy-making generation model."""
  
  def __init__(self, params):
    super(BaseAnalogyModel, self).__init__(params)

  def get_inputs_from_placeholder(self, dataset_size, batch_size):
    params = self._params
    assert params.min_input_length == params.max_input_length
    init_length = params.max_input_length
    sample_length = init_length + params.max_length
    #
    placeholder = dict()
    placeholder['AB_landmarks'] = tf.placeholder(
      dtype=tf.float32, shape=(batch_size, sample_length, params.keypoint_dim, 2))
    placeholder['CD_landmarks'] = tf.placeholder(
      dtype=tf.float32, shape=(batch_size, sample_length, params.keypoint_dim, 2))
    #
    inputs = dict()
    inputs.update(placeholder)
    inputs['dataset_size'] = dataset_size
    inputs['A_lens'] = tf.tile([init_length], [batch_size])
    inputs['B_lens'] = tf.tile([params.max_length], [batch_size])
    inputs['C_lens'] = tf.tile([init_length], [batch_size])
    inputs['D_lens'] = tf.tile([params.max_length], [batch_size])
    inputs['transform_ratio'] = tf.ones([batch_size, 1], dtype=tf.float32)
    return inputs

  def get_duplicate_inputs(self, dataset_dir, dataset_name, split_name,
                           batch_size, load_image):
    params = self._params
    init_length = params.max_input_length
    sample_length = init_length + params.max_length
    #
    with tf.variable_scope('data_loading_%s/%s' % (dataset_name, split_name)):
      raw_inputs = input_generator.get_tfrecord(
        dataset_dir, dataset_name, split_name,
        shuffle=True, sample_length=sample_length)
      raw_inputs = _get_data_from_provider(
        raw_inputs, batch_size, split_name,
        is_training=False, load_image=load_image)
      #
      assert batch_size > 1
      #assert params.eval_mode == 'analogy'
      inputs = dict()
      inputs['dataset_size'] = raw_inputs['dataset_size']
      #
      inputs['AB_landmarks'] = tf.tile(
        raw_inputs['landmarks'][0:1], [batch_size, 1, 1, 1])
      inputs['CD_landmarks'] = tf.tile(
        raw_inputs['landmarks'][1:2], [batch_size, 1, 1, 1])
      inputs['transform_ratio'] = tf.ones(
        [batch_size, 1], dtype=tf.float32)
      #
      inputs['A_lens'] = tf.tile([init_length], [batch_size])
      inputs['B_lens'] = tf.tile([params.max_length], [batch_size])
      inputs['C_lens'] = tf.tile([init_length], [batch_size])
      inputs['D_lens'] = tf.tile([params.max_length], [batch_size])
    return inputs

  def preprocess(self, raw_inputs, is_training, load_image=False):
    """Split Data."""
    params = self._params
    shp = raw_inputs['AB_landmarks'].get_shape().as_list()
    quantity, max_length = shp[0], shp[1]
    inputs = dict()
    inputs['dataset_size'] = raw_inputs['dataset_size']
    inputs['A_lens'] = raw_inputs['A_lens']
    inputs['B_lens'] = raw_inputs['B_lens']
    inputs['C_lens'] = raw_inputs['C_lens']
    inputs['D_lens'] = raw_inputs['D_lens']
    inputs['transform_ratio'] = raw_inputs['transform_ratio']
    #
    inputs['A_landmarks'], inputs['B_landmarks'] = \
      input_generator.split_pts_seq(
        raw_inputs['AB_landmarks'], inputs['A_lens'], inputs['B_lens'],
        params.max_input_length, params.max_length, is_training)
    inputs['C_landmarks'], inputs['D_landmarks'] = \
      input_generator.split_pts_seq(
        raw_inputs['CD_landmarks'], inputs['C_lens'], inputs['D_lens'],
        params.max_input_length, params.max_length, is_training)
    return inputs

  def get_model_fn(self, is_training, reuse, output_length=None):
    model_fn = model_factory.get_model_fn(
      self._params, is_training, reuse, output_length)
    return model_fn

  def save_html_page(self, save_dir, num_repeat, batch_size):
    content = '<html><body><h1>Visualization</h1><table boder="1" style="width=100%">'
    content = content + '<tr><td>Sequence A</td><td>Sequence A to B</td><td>Sequence C</td><td>Sequence C to D</td><td>Generated Sequence C to D (multiple trajectories)</td>'
    for t in xrange(num_repeat):
      content += '\n<tr>'
      content += '<td><img src=\"%02d_A.gif\" style=\"width:255px;height:255px;\"></td>' % t
      content += '<td><img src=\"%02d_AB.gif\" style=\"width:255px;height:255px;\"></td>' % t
      content += '<td><img src=\"%02d_C.gif\" style=\"width:255px;height:255px;\"></td>' % t
      content += '<td><img src=\"%02d_CD.gif\" style=\"width:255px;height:255px;\"></td>' % t
      content += '<td><img src=\"%02d_pred.gif\" style=\"width:%dpx;height:255px;\"></td>' % (t, batch_size * 255)
      content += '</tr>'
    content += '</html>'
    with open(os.path.join(save_dir, 'index.html'), 'w') as f:
      f.write(content)
      f.flush()
    f.close()

  def get_visualization_op(self, inputs, outputs, log_dir, counter, output_length=None):
    params = self._params
    batch_size = params.batch_size
    max_length = params.max_length
    img_size = params.img_size
    
    A_lens = inputs['A_lens']
    B_lens = inputs['B_lens']
    C_lens = inputs['C_lens']
    D_lens = inputs['D_lens']
    if output_length is None:
      pred_lens = D_lens
    else:
      pred_lens = tf.zeros_like(D_lens, dtype=tf.int32) + output_length
    A_landmarks = inputs['A_landmarks'] * img_size
    B_landmarks = inputs['B_landmarks'] * img_size
    C_landmarks = inputs['C_landmarks'] * img_size
    D_landmarks = inputs['D_landmarks'] * img_size
    pred_landmarks = outputs['D_landmarks'] * img_size

    def write_grid(base_id, A_lens, B_lens, C_lens, D_lens, pred_lens,
                   A_landmarks, B_landmarks, C_landmarks, D_landmarks, pred_landmarks):
      """Python function."""
      #
      video_proc_lib = VideoProc()
      A_video = np.zeros((A_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(A_lens[0]):
        A_video[t] = utils.visualize_landmarks(A_video[t], A_landmarks[0][t]) 
        A_video[t] = utils.visualize_h36m_skeleton(A_video[t], A_landmarks[0][t])
        A_video[t] = utils.visualize_boundary(A_video[t], colormap='green')
      B_video = np.zeros((B_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(B_lens[0]):
        B_video[t] = utils.visualize_landmarks(B_video[t], B_landmarks[0][t])
        B_video[t] = utils.visualize_h36m_skeleton(B_video[t], B_landmarks[0][t])
        B_video[t] = utils.visualize_boundary(B_video[t], colormap='blue')
      merged_video = np.concatenate((A_video, B_video), axis=0)
      video_proc_lib.save_img_seq_to_video(
        A_video, log_dir, '%02d_A.gif' % base_id, frame_rate=7.5, codec=None, override=True)
      video_proc_lib.save_img_seq_to_video(
        merged_video, log_dir, '%02d_AB.gif' % base_id, frame_rate=7.5, codec=None, override=True) 
      #
      C_video = np.zeros((C_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(C_lens[0]):
        C_video[t] = utils.visualize_landmarks(C_video[t], C_landmarks[0][t])
        C_video[t] = utils.visualize_h36m_skeleton(C_video[t], C_landmarks[0][t])
        C_video[t] = utils.visualize_boundary(C_video[t], colormap='green')
      D_video = np.zeros((D_lens[0], img_size, img_size, 3), dtype=np.float32)
      for t in xrange(D_lens[0]):
        D_video[t] = utils.visualize_landmarks(D_video[t], D_landmarks[0][t])
        D_video[t] = utils.visualize_h36m_skeleton(D_video[t], D_landmarks[0][t])
        D_video[t] = utils.visualize_boundary(D_video[t], colormap='blue')
      merged_video = np.concatenate((C_video, D_video), axis=0)
      video_proc_lib.save_img_seq_to_video(
        C_video, log_dir, '%02d_C.gif' % base_id, frame_rate=7.5, codec=None, override=True)
      video_proc_lib.save_img_seq_to_video(
        merged_video, log_dir, '%02d_CD.gif' % base_id, frame_rate=7.5, codec=None, override=True)
      #
      raw_gif_list = []
      for i in xrange(batch_size):
        print(i)
        pred_video = np.zeros((pred_lens[i], img_size, img_size, 3), dtype=np.float32)
        for t in xrange(pred_lens[i]):
          pred_video[t] = utils.visualize_landmarks(pred_video[t], pred_landmarks[i][t])
          pred_video[t] = utils.visualize_h36m_skeleton(pred_video[t], pred_landmarks[i][t])
          pred_video[t] = utils.visualize_boundary(pred_video[t], colormap='red')
        merged_video = np.concatenate((C_video, pred_video), axis=0)
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
      inp=[counter, A_lens, B_lens, C_lens, D_lens, pred_lens,
           A_landmarks, B_landmarks, C_landmarks, D_landmarks, pred_landmarks],
      Tout=[tf.int64], name='write_grid')[0]
    return save_op
