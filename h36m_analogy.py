"""Running analogy-making on Human3.6M."""

import os
import numpy as np
import tensorflow as tf

from tensorflow import app
from H36M_MTVAEAnalogyModel import MTVAEAnalogyModel

import h36m_input as input_generator

import utils
from preprocess.video_proc_utils import VideoProc
from H36M_testfile import H36M_analogy as H36M_testcases
from H36M_modelfile import *

flags = tf.app.flags
slim = tf.contrib.slim

flags.DEFINE_string('inp_dir', 'workspace/', '')
flags.DEFINE_string('dataset_name', 'Human3.6M', '')
flags.DEFINE_string('model_version', 'MTVAE', '')
flags.DEFINE_string('checkpoint_dir', 'checkpoints/', '')
flags.DEFINE_string('master', '', '')
flags.DEFINE_integer('min_input_length', 16, '')
flags.DEFINE_integer('max_input_length', 16, '')
flags.DEFINE_integer('max_length', 32, '')
# Model paramters.
flags.DEFINE_integer('keypoint_dim', 32, '')
flags.DEFINE_integer('img_size', 180, '')
flags.DEFINE_integer('batch_size', 1, '')
flags.DEFINE_string('enc_model', 'ln_lstm', '')
flags.DEFINE_string('dec_model', 'ln_lstm', '')
flags.DEFINE_integer('use_bidirection_lstm', 0, '')
flags.DEFINE_integer('use_recurrent_dropout', 0, '')
flags.DEFINE_float('recurrent_dropout_prob', 1.0, '')
# Sample parameters.
flags.DEFINE_integer('use_prior', 0, '')
flags.DEFINE_integer('sample_pdf', 1, '')
flags.DEFINE_float('sample_temp', 0.1, '')

FLAGS = flags.FLAGS

SEQ_LEN = 600
HIS_KEYFRAMES = [0, 7, 15]
FUT_KEYFRAMES = [0, 7, 15, 23, 31]

MODEL_SPECS = {
  #'VAE': VAE,
  'MTVAE': MTVAE,
}

MODEL_TO_NAME = {
  #'VAE': 'Vanill VAE',
  'MTVAE': 'Motion Transformation VAE',
}

MODEL_TO_CLASS = {
  #'VAE': VAEAnalogyModel,
  'MTVAE': MTVAEAnalogyModel,
}

MODEL_TO_SCOPE = {
  #'VAE': ['seq_enc', 'fut_dec', 'latent_enc', 'latent_dec'],
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
  vid_ids = []
  base_frame_ids = []
  assert dataset_name == 'Human3.6M'
  for i in H36M_testcases.keys():
    value_AB = H36M_testcases[i][0]
    value_CD = H36M_testcases[i][1]
    # Add AB.
    filenames.append('%s/%s' % (value_AB[0], value_AB[1]))
    frame_nums.append(int(value_AB[2]))
    base_frame = int(value_AB[1][-3:]) * SEQ_LEN
    vid_ids.append(int(value_AB[0]))
    base_frame_ids.append(base_frame)
    # Add CD.
    filenames.append('%s/%s' % (value_CD[0], value_CD[1]))
    frame_nums.append(int(value_CD[2]))
    base_frame = int(value_CD[1][-3:]) * SEQ_LEN
    vid_ids.append(int(value_CD[0]))
    base_frame_ids.append(base_frame)
    #
  return filenames, frame_nums, vid_ids, base_frame_ids

def get_dataset(dataset_dir, dataset_name):
  #image_dir = 'raw_images'
  pts_dir = 'annot_pts'
  filenames, frame_nums, vid_ids, base_frame_ids = \
    get_filenames(dataset_dir, dataset_name)
  annot_pts = [os.path.join(dataset_dir, dataset_name, pts_dir, '%s.csv' % fp) for fp in filenames]
  ############################
  ## Prepare Filename Queue ##
  ############################
  filename_queue = [annot_pts, frame_nums, vid_ids, base_frame_ids]
  return filename_queue

def save_html_page(img_dir, quantity):
  for t in xrange(quantity):
    content = '<html><body><h1>Analogy-making Results: D = (B-A)+C</h1><table border="1">'
    content = content + '<tr><td>Method</td><td>Keypoint sequence</td></tr>'
    for nick_name in MODEL_SPECS.keys():
      name = MODEL_TO_NAME[nick_name]
      sub_name = '%02d_%s' % (t, nick_name)
      content += '\n<tr><td>%s Sequence A (input)</td>' % name
      content += '<td><img src=\"%s_A.png\" style=\"width:300px;height:100px;\"></td></tr>' % sub_name
      
      content += '\n<tr><td>%s Sequence B (input)</td>' % name
      content += '<td><img src=\"%s_B.png\" style=\"width:500px;height:100px;\"></td></tr>' % sub_name
      
      content += '\n<tr><td>%s Sequence C (input)</td>' % name
      content += '<td><img src=\"%s_C.png\" style=\"width:300px;height:100px;\"></td></tr>' % sub_name
      
      content += '\n<tr><td>%s Sequence D (output)</td>' % name
      content += '<td><img src=\"%s_predD.png\" style=\"width:500px;height:100px;\"></td></tr>' % sub_name
    content += '</html>'
    with open(os.path.join(img_dir, '%02d.html' % t), 'w') as f:
      f.write(content)
      f.flush()
    f.close()

def save_images(img_dir, img_idx, file_suffix, inp_dict):
  A_imgs = utils.flatten_img_seq(inp_dict['A_imgs'])
  B_imgs = utils.flatten_img_seq(inp_dict['B_imgs'])
  C_imgs = utils.flatten_img_seq(inp_dict['C_imgs'])
  predD_imgs = utils.flatten_img_seq(inp_dict['predD_imgs'])
  sub_name = '%02d_%s' % (img_idx, file_suffix)
  utils.save_image(A_imgs, os.path.join(img_dir, '%s_A.png' % sub_name))
  utils.save_image(B_imgs, os.path.join(img_dir, '%s_B.png' % sub_name))
  utils.save_image(C_imgs, os.path.join(img_dir, '%s_C.png' % sub_name))
  utils.save_image(predD_imgs, os.path.join(img_dir, '%s_predD.png' % sub_name))

def run_visualization(video_proc_lib, video_dir, video_idx, file_suffix,
                      A_imgs, B_imgs, C_imgs, predD_imgs):
  A_imgs, B_imgs = np.copy(A_imgs), np.copy(B_imgs)
  Cimgs, predD_imgs = np.copy(C_imgs), np.copy(predD_imgs)

  his_len = A_imgs.shape[0]
  fut_len = B_imgs.shape[0]
  for t in xrange(his_len):  
    A_imgs[t] = utils.visualize_boundary(A_imgs[t], radius=4, colormap='green')
    C_imgs[t] = utils.visualize_boundary(C_imgs[t], radius=4, colormap='purple')
  for t in xrange(fut_len):
    B_imgs[t] = utils.visualize_boundary(B_imgs[t], radius=4, colormap='blue')
    predD_imgs[t] = utils.visualize_boundary(predD_imgs[t], radius=4, colormap='red')

  AB_imgs_left = np.concatenate((A_imgs, np.zeros_like(B_imgs)), axis=0)
  AB_imgs_right = np.concatenate((A_imgs, B_imgs), axis=0)
  AB_imgs = np.concatenate((AB_imgs_left, AB_imgs_right), axis=2)

  CD_imgs_left = np.concatenate((C_imgs, np.zeros_like(predD_imgs)), axis=0)  
  CD_imgs_right = np.concatenate((C_imgs, predD_imgs), axis=0)
  CD_imgs = np.concatenate((CD_imgs_left, CD_imgs_right), axis=2)

  out_dict = dict()
  out_dict['A_imgs'] = np.copy(A_imgs[HIS_KEYFRAMES])
  out_dict['B_imgs'] = np.copy(B_imgs[FUT_KEYFRAMES])
  out_dict['C_imgs'] = np.copy(C_imgs[HIS_KEYFRAMES])
  out_dict['predD_imgs'] = np.copy(predD_imgs[FUT_KEYFRAMES])
  ##################
  ## merge videos ##
  ##################
  merged_video = np.concatenate((AB_imgs, CD_imgs), axis=2)

  video_file = '%02d_merged_%s.gif' % (video_idx, file_suffix)
  video_proc_lib.save_img_seq_to_video(merged_video, video_dir, video_file,
    frame_rate=7.5, codec='', override=True)
  return video_file, out_dict


def main(_):
  params = add_attributes(FLAGS, MODEL_SPECS[FLAGS.model_version])
  model_dir = os.path.join(params.checkpoint_dir, params.model_name, 'train')
  img_dir = os.path.join(params.checkpoint_dir, 'analogy_comparison', 'imgs')
  log_dir = os.path.join(params.checkpoint_dir, 'analogy_comparison', params.model_version)
  assert os.path.isdir(model_dir)
  utils.force_mkdir(os.path.join(params.checkpoint_dir, 'analogy_comparison'))
  utils.force_mkdir(log_dir)
  utils.force_mkdir(img_dir)
  video_proc_lib = VideoProc()
  ##################
  ## Load dataset ##
  ##################
  filename_queue = get_dataset(FLAGS.inp_dir, FLAGS.dataset_name)
  dataset_size = len(filename_queue[0])
  assert params.min_input_length == params.max_input_length
  init_length = params.min_input_length
  sample_length = params.max_length + init_length

  np_data = dict()
  np_data['landmarks'] = np.zeros(
    (dataset_size, sample_length, params.keypoint_dim, 2), dtype=np.float32)
  np_data['actual_frame_id'] = []
  np_data['vid_id'] = []
  for i in xrange(dataset_size):
    mid_frame = int(filename_queue[1][i])
    keyframes = np.arange(mid_frame - init_length, mid_frame + params.max_length) 
    _, landmarks = input_generator.load_pts_seq(
      params.dataset_name, filename_queue[0][i], sample_length, keyframes)
    landmarks = np.reshape(
      landmarks, (sample_length, params.keypoint_dim, 2))
    np_data['landmarks'][i] = landmarks
    # TODO(xcyan): hacky implemetation.
    np_data['vid_id'].append(filename_queue[2][i])
    np_data['actual_frame_id'].append(filename_queue[3][i] + mid_frame)
  ####################
  ## save html page ##
  ####################
  save_html_page(img_dir, dataset_size/2)
  #########################
  ## Prepare output dict ##
  #########################
  genseq = dict()
  genseq['AB_id'] = []
  genseq['CD_id'] = []
  genseq['A_start'] = []
  genseq['B_start'] = []
  genseq['C_start'] = []
  genseq['pred_start'] = []
  genseq['pred_keypoints'] = []
  ####################
  ## Build tf.graph ##
  ####################
  g = tf.Graph()
  with g.as_default():
    keypointClass = MODEL_TO_CLASS[params.model_version]
    model = keypointClass(params)
    scopes = MODEL_TO_SCOPE[params.model_version]

    eval_data = model.get_inputs_from_placeholder(dataset_size, params.batch_size)
    inputs = model.preprocess(eval_data, is_training=False, load_image=False)
    assert (not params.use_prior)
    model_fn = model.get_model_fn(is_training=False, reuse=False)
    outputs = model_fn(inputs)
    variables_to_restore = slim.get_variables_to_restore(scopes)
    #######################
    ## Restore variables ##
    #######################
    checkpoint_path = tf.train.latest_checkpoint(model_dir)
    restorer = tf.train.Saver(variables_to_restore)
    ####################
    ## launch session ##
    ####################
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())
      restorer.restore(sess, checkpoint_path)

      for i in xrange(dataset_size/2):
        print(i)
        AB_idx, CD_idx = i * 2, i * 2 + 1 
        A_lms, B_lms, C_lms, predD_lms = sess.run(
          [inputs['A_landmarks'], inputs['B_landmarks'],
           inputs['C_landmarks'], outputs['D_landmarks']],
          feed_dict={
            eval_data['AB_landmarks']: np_data['landmarks'][AB_idx:AB_idx+1],
            eval_data['CD_landmarks']: np_data['landmarks'][CD_idx:CD_idx+1]})
        batch_output = np.copy(predD_lms) * 2 - 1.0  
        A_imgs = utils.visualize_h36m_skeleton_batch(
          np.copy(A_lms[0]), params.img_size)
        B_imgs = utils.visualize_h36m_skeleton_batch(
          np.copy(B_lms[0]), params.img_size)
        C_imgs = utils.visualize_h36m_skeleton_batch(
          np.copy(C_lms[0]), params.img_size)
        predD_imgs = utils.visualize_h36m_skeleton_batch(
          np.copy(predD_lms[0]), params.img_size)
        ################
        ## Save video ##
        ################
        vid_file, out_dict = run_visualization(video_proc_lib, log_dir, i,
          params.model_version, A_imgs, B_imgs, C_imgs, predD_imgs)
        save_images(img_dir, i, params.model_version, out_dict)
        ##########
        ## save ##
        ##########
        genseq['AB_id'].append(np_data['vid_id'][AB_idx])
        genseq['CD_id'].append(np_data['vid_id'][CD_idx])
        genseq['A_start'].append(np_data['actual_frame_id'][AB_idx] - init_length)
        genseq['B_start'].append(np_data['actual_frame_id'][AB_idx])
        genseq['C_start'].append(np_data['actual_frame_id'][CD_idx] - init_length)
        genseq['pred_start'].append(np_data['actual_frame_id'][CD_idx])
        genseq['pred_keypoints'].append(batch_output)
        print('%g %g' % (np.amin(batch_output[:, :, :, 0]), np.amax(batch_output[:, :, :, 0])))
        print('%g %g' % (np.amin(batch_output[:, :, :, 1]), np.amax(batch_output[:, :, :, 1])))
      
  utils.save_python_objects(genseq, os.path.join(log_dir, 'test_analogy.pkl'))


if __name__ == '__main__':
  app.run()  
