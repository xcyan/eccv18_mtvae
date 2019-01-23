"""Generate Keypoint Sequences for evaluation/visualization purposes."""

import os
import numpy as np
import tensorflow as tf

from tensorflow import app
from H36M_VAEPredModel import VAEPredModel
from H36M_MTVAEPredModel import MTVAEPredModel

import h36m_input as input_generator

import utils
import pickle
import csv
from preprocess.video_proc_utils import VideoProc
from H36M_testfile import H36M_generation as H36M_testcases
from H36M_modelfile import *

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir', 'workspace/', '')
flags.DEFINE_string('dataset_name', 'Human3.6M', '')
flags.DEFINE_string('model_version', 'MTVAE', '')
flags.DEFINE_string('master', '', '')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/', '')
flags.DEFINE_integer('min_input_length', 16, '')
flags.DEFINE_integer('max_input_length', 16, '')
flags.DEFINE_integer('max_length', 64, '')
flags.DEFINE_integer('keypoint_dim', 32, '')
flags.DEFINE_integer('img_size', 180, '')
flags.DEFINE_integer('batch_size', 5, '')
# model parameters.
flags.DEFINE_string('enc_model', 'ln_lstm', '')
flags.DEFINE_string('dec_model', 'ln_lstm', '')
flags.DEFINE_integer('use_bidirection_lstm', 0, '')
flags.DEFINE_integer('use_recurrent_dropout', 0, '')
flags.DEFINE_float('recurrent_dropout_prob', 1.0, '')
# Sample parameters.
flags.DEFINE_integer('use_prior', 1, '')
flags.DEFINE_integer('sample_pdf', 1, '')
flags.DEFINE_float('sample_temp', 0.1, '')

FLAGS = flags.FLAGS

SEQ_LEN = 600
HIS_KEYFRAMES = [0, 7, 15]
FUT_KEYFRAMES = [0, 3, 15, 31, 63]

MODEL_SPECS = {
  'VAE': VAE,
  'MTVAE': MTVAE,
}

MODEL_TO_NAME = {
  'VAE': 'Vanill VAE',
  'MTVAE': 'Motion Transformation VAE',
}

MODEL_TO_CLASS = {
  'VAE': VAEPredModel,
  'MTVAE': MTVAEPredModel,
}

MODEL_TO_SCOPE = {
  'VAE': ['seq_enc', 'fut_dec', 'latent_enc', 'latent_dec'],
  'MTVAE': ['seq_enc', 'fut_dec', 'latent_enc', 'latent_dec'],
}

def add_attributes(input_params, dicts):
  output_params = input_params
  for k in dicts.keys():
    setattr(output_params, k, dicts[k])
  return output_params

def get_filenames(dataset_dir, dataset_name):
  filenames = []
  frame_nums = []
  assert dataset_name == 'Human3.6M'
  # vid_id, vid_fullname, frame_number, actual_frame_number
  for i in H36M_testcases.keys():
    vid_id = H36M_testcases[i][0]
    vid_filename = H36M_testcases[i][1]
    frame_id = H36M_testcases[i][2]
    #
    filenames.append('%s/%s' % (vid_id, vid_filename))
    frame_nums.append(int(frame_id))
  return filenames, frame_nums

def get_dataset(dataset_dir, dataset_name):
  pts_dir = 'annot_pts'
  filenames, frame_nums = get_filenames(dataset_dir, dataset_name)
  annot_pts = [os.path.join(dataset_dir, dataset_name, pts_dir, '%s.csv' % fp) for fp in filenames]
  ############################
  ## Prepare Filename Queue ##
  ############################
  filename_queue = [annot_pts, frame_nums]
  return filename_queue

def save_images(img_dir, img_idx, file_suffix, inp_dict):
  his_imgs = utils.flatten_img_seq(inp_dict['his_imgs'])
  fut_imgs = utils.flatten_img_seq(inp_dict['fut_imgs'])
  #
  if file_suffix == 'gt':
    utils.save_image(his_imgs, os.path.join(img_dir, '%02d_%s_his_poses.png' % (img_idx, file_suffix)))
  utils.save_image(fut_imgs, os.path.join(img_dir, '%02d_%s_fut_poses.png' % (img_idx, file_suffix)))
 

def run_visualization(video_proc_utils, video_dir, video_idx, file_suffix,
      his_imgs, fut_imgs):
  his_imgs = np.copy(his_imgs)
  fut_imgs = np.copy(fut_imgs)
  video_file = '%02d_%s.gif' % (video_idx, file_suffix)
  #########################
  ## Adding bounding box ##
  #########################
  his_len = his_imgs.shape[0]
  fut_len = fut_imgs.shape[0]
  for t in xrange(his_len):
    his_imgs[t] = utils.visualize_boundary(his_imgs[t], radius=4, colormap='green')
  for t in xrange(fut_len):
    fut_imgs[t] = utils.visualize_boundary(fut_imgs[t], radius=4, colormap='red')
  out_dict = dict()
  out_dict['his_imgs'] = np.copy(his_imgs[HIS_KEYFRAMES])
  out_dict['fut_imgs'] = np.copy(fut_imgs[FUT_KEYFRAMES])
  ##################
  ## merge videos ##
  ################## 
  merged_video = np.concatenate((his_imgs, fut_imgs), axis=0)
  video_proc_utils.save_img_seq_to_video(merged_video, video_dir,
    video_file, frame_rate=7.5, codec='', override=True)
  return video_file, out_dict


def main(_):
  params = add_attributes(FLAGS, MODEL_SPECS[FLAGS.model_version])
  model_dir = os.path.join(FLAGS.checkpoint_dir, params.model_name, 'train')
  img_dir = os.path.join(params.checkpoint_dir, 'gensample_comparison', 'imgs')
  log_dir = os.path.join(params.checkpoint_dir, 'gensample_comparison', params.model_version)
  if (params.model_version in ['PredLSTM']):
    params.batch_size = 1
  assert os.path.isdir(model_dir)
  utils.force_mkdir(os.path.join(params.checkpoint_dir, 'gensample_comparison'))
  utils.force_mkdir(log_dir)
  utils.force_mkdir(img_dir)
  video_proc_utils = VideoProc()
  ##################
  ## load dataset ##
  ##################
  filename_queue = get_dataset(params.inp_dir, params.dataset_name)
  dataset_size = len(filename_queue[0])
  assert params.min_input_length == params.max_input_length
  init_length = params.min_input_length
  sample_length = params.max_length + init_length
  #
  np_data = dict()
  np_data['vid_id'] = []
  np_data['actual_frame_id'] = []
  np_data['landmarks'] = np.zeros(
    (dataset_size, sample_length, params.keypoint_dim, 2), dtype=np.float32)
  for i in xrange(dataset_size):
    mid_frame = int(filename_queue[1][i])
    keyframes = np.arange(mid_frame - init_length, mid_frame + params.max_length)
    _, landmarks = input_generator.load_pts_seq(
      params.dataset_name, filename_queue[0][i], sample_length, keyframes)
    landmarks = np.reshape(
      landmarks, (sample_length, params.keypoint_dim, 2))
    np_data['landmarks'][i] = landmarks
    # TODO(xcyan): hacky implementation.
    base_frame_id = int(H36M_testcases[i][1][-3:]) * SEQ_LEN
    np_data['vid_id'].append(int(H36M_testcases[i][0]))
    np_data['actual_frame_id'].append(base_frame_id + mid_frame)
  #########################
  ## Prepare output dict ##
  #########################
  genseq = dict()
  genseq['video_id'] = []
  genseq['ob_start'] = []
  genseq['pred_start'] = []
  genseq['pred_keypoints'] = []
  #################
  ## Build graph ##
  #################
  g = tf.Graph()
  with g.as_default():
    keypointClass = MODEL_TO_CLASS[params.model_version]
    model = keypointClass(params)
    scopes = MODEL_TO_SCOPE[params.model_version]

    eval_data = model.get_inputs_from_placeholder(
      dataset_size, params.batch_size)
    inputs = model.preprocess(
      eval_data, is_training=False, load_image=False)
    ##############
    ## model_fn ##
    ##############
    if not (params.model_version in ['PredLSTM']):
      model_fn = model.get_sample_fn(
        is_training=False, use_prior=params.use_prior, reuse=False) 
    else:
      model_fn = model.get_sample_fn(is_training=False, reuse=False)
    outputs = model_fn(inputs)
    #######################
    ## restore variables ##
    #######################
    variables_to_restore = slim.get_variables_to_restore(scopes)
    restorer = tf.train.Saver(variables_to_restore)
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    ####################
    ## launch session ##
    ####################
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      restorer.restore(sess, checkpoint_path)
      for i in xrange(dataset_size):
        print(i)
        raw_gif_list = []
        curr_landmarks = np.tile(
          np_data['landmarks'][i:i+1], (params.batch_size, 1, 1, 1))
        his_lms, fut_lms, pred_lms = sess.run(
          [inputs['his_landmarks'], inputs['fut_landmarks'], outputs['fut_landmarks']],
          feed_dict={
            eval_data['landmarks']: curr_landmarks})

        batch_output = np.copy(pred_lms) * 2 - 1.0
        for run_id in xrange(params.batch_size):
          his_imgs = utils.visualize_h36m_skeleton_batch(
            np.copy(his_lms[run_id]), params.img_size)
          fut_imgs = utils.visualize_h36m_skeleton_batch(
            np.copy(fut_lms[run_id]), params.img_size)
          pred_imgs = utils.visualize_h36m_skeleton_batch(
            np.copy(pred_lms[run_id]), params.img_size)
          ################
          ## save video ##
          ################
          if run_id == 0:
            vid_file, out_dict = run_visualization(video_proc_utils, log_dir, i, 
              'gt', his_imgs, fut_imgs)
            save_images(img_dir, i, 'gt', out_dict)

          vid_file, out_dict = run_visualization(video_proc_utils, log_dir, i,
            params.model_version + '_%02d' % run_id, his_imgs, pred_imgs)
          save_images(img_dir, i, params.model_version + '_%02d' % run_id, out_dict)
          raw_gif_list.append(vid_file)
        #################
        ## Merge video ##
        #################
        if params.batch_size > 1:
          video_proc_utils.merge_video_side_by_side(
            log_dir, raw_gif_list, '%02d_merged_%s.gif' % (i, params.model_version), override=True)
        ##########
        ## Save ##
        ##########
        genseq['video_id'].append(np_data['vid_id'][i])
        genseq['ob_start'].append(np_data['actual_frame_id'][i] - init_length)
        genseq['pred_start'].append(np_data['actual_frame_id'][i]) 
        genseq['pred_keypoints'].append(batch_output)
        print('%g %g' % (np.amin(batch_output[:, :, :, 0]), np.amax(batch_output[:, :, :, 0])))
        print('%g %g' % (np.amin(batch_output[:, :, :, 1]), np.amax(batch_output[:, :, :, 1])))
         

  utils.save_python_objects(genseq, os.path.join(log_dir, 'test.pkl'))


if __name__ == '__main__':
  app.run()

