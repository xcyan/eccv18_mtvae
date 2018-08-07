"""Provides input dictionaries for Human3.6M keypoint experiments."""

import os
import csv
from scipy.io import loadmat
import tensorflow as tf
import numpy as np
import time
import random
import utils

DATASET_TO_METADATA = {
  'Human3.6M': {
    'pts_dir': 'annot_pts',
    'image_dir': 'raw_images',
    'im_height': 180,
    'im_width': 180,
    'keypoint_dim': 32, 
  },
}

def real_path(symbol_path):
  return os.path.realpath(symbol_path)

def get_filenames(dataset_dir, dataset_name, split_name):
  with open(os.path.join(dataset_dir, dataset_name, '%s_files.csv' %  split_name), 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    filenames = []
    for line in csvreader:
      filenames.append('%s/%s' % (line[0], line[1]))
  return filenames

def get_dataset(dataset_dir, dataset_name, split_name):
  """Provides filename_queue from a specific (dataset, split)."""
  metadata = DATASET_TO_METADATA[dataset_name]
  pts_dir = metadata['pts_dir']
  image_dir = metadata['image_dir']
  #
  filenames = get_filenames(dataset_dir, dataset_name, split_name)
  annot_pts = [os.path.join(dataset_dir, dataset_name, pts_dir, '%s.csv' % fp) for fp in filenames]
  image_list = [os.path.join(dataset_dir, dataset_name, image_dir, '%s_*.jpg' % fp) for fp in filenames]
  #############################
  ## Preppare Filename Queue ##
  #############################
  filename_queue = [annot_pts, image_list]
  return filename_queue

def load_pts_seq(dataset_name, filename, max_length, keyframes=None):
  metadata = DATASET_TO_METADATA[dataset_name]
  keypoint_dim = metadata['keypoint_dim']
  im_height = metadata['im_height']
  im_width = metadata['im_width']
  # Read CSV contents.
  csv_contents = []
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      csv_contents.append(row)
  csvfile.close()
  assert csv_contents[0][0] == 'Number frames'
  # TODO(xcyan): fix this 
  seq_len = int(csv_contents[0][1])-1
  pts_seq = np.zeros((seq_len, keypoint_dim, 2), dtype=np.float32)
  for row_index in xrange(keypoint_dim):
    row = csv_contents[row_index+1]
    px = np.asarray(row, dtype=np.float32)[0:seq_len*2:2]
    py = np.asarray(row, dtype=np.float32)[1:seq_len*2:2]
    px = np.reshape(px, (1, 1, seq_len))
    py = np.reshape(py, (1, 1, seq_len))
    pts_seq[0:seq_len, row_index, 0] = px / float(im_height)
    pts_seq[0:seq_len, row_index, 1] = py / float(im_width)
  #
  if keyframes is not None:
    assert keyframes.shape[0] <= max_length
    pts_seq = np.take(pts_seq, keyframes, axis=0)

  if max_length > seq_len:
    pts_seq = np.concatenate(
      [pts_seq, np.zeros((max_length-seq_len, keypoint_dim, 2), dtype=np.float32)], axis=0)
  
  return seq_len, pts_seq


def sample_pts_seq(dataset_name, filename, max_length, skip_n_frame):
  """Python wrapper that loads keypoint sequence."""
  metadata = DATASET_TO_METADATA[dataset_name]
  keypoint_dim = metadata['keypoint_dim']
  im_height = metadata['im_height']
  im_width = metadata['im_width']
  # Read CSV contents.
  csv_contents = []
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    for row in csvreader:
      csv_contents.append(row)
  csvfile.close()
  assert csv_contents[0][0] == 'Number frames'
  video_length = int(csv_contents[0][1])
  keyframes = np.zeros((max_length), dtype=np.int32)
  start_idx = random.randint(0, video_length-max_length*skip_n_frame-1)
  # Read.
  for t in xrange(max_length):
    keyframes[t] = int(start_idx + t * skip_n_frame)
  #
  pts_seq = np.zeros((max_length, keypoint_dim, 2), dtype=np.float32)
  for row_index in xrange(keypoint_dim):
    row = csv_contents[row_index+1]
    px = np.asarray(row, dtype=np.float32)[keyframes * 2]
    py = np.asarray(row, dtype=np.float32)[keyframes * 2 + 1]
    px = np.reshape(px, (1, 1, max_length))
    py = np.reshape(py, (1, 1, max_length))
    pts_seq[:, row_index, 0] = px / float(im_height)
    pts_seq[:, row_index, 1] = py / float(im_width)
  return keyframes, pts_seq

def sample_image_seq(dataset_name, filename_pattern, max_length, keyframes):
  metadata = DATASET_TO_METADATA[dataset_name]
  im_height = metadata['im_height']
  im_width = metadata['im_width']
  image_seq = np.zeros((max_length, im_height, im_width, 3), dtype=np.float32)
  assert (keyframes.shape[0] == max_length)
  #print('loading images: %s' % filename_pattern)
  for i in xrange(max_length):
    #print('loading images [%02d]: %s' % (i, filename_pattern))
    image_seq[i] = utils.load_image(filename_pattern.replace('*', '%05d' % keyframes[i]))
  return image_seq

def preprocess_data_split(source_data, his_lens, fut_lens,
                          max_input_length, max_output_length):
  quantity = source_data.shape[0]
  his_data = np.zeros_like(source_data, dtype=np.float32)
  fut_data = np.zeros_like(source_data, dtype=np.float32)
  #
  for i in xrange(quantity):
    his_data[i, 0:his_lens[i]] = source_data[i, 0:his_lens[i]]
    fut_data[i, 0:fut_lens[i]] = source_data[i, his_lens[i]:his_lens[i]+fut_lens[i]]
  #
  his_data = his_data[:, 0:max_input_length]
  fut_data = fut_data[:, 0:max_output_length]
  return his_data, fut_data

def split_pts_seq(source_pts, his_lens, fut_lens, 
                  max_input_length, max_output_length, data_aug):
  shp = source_pts.get_shape().as_list()
  quantity, keypoint_dim = shp[0], shp[2]
  # Apply pyfunc.
  his_pts, fut_pts = tf.py_func(preprocess_data_split,
    [source_pts, his_lens, fut_lens, max_input_length, max_output_length],
    [tf.float32, tf.float32], name='split_pts_seq')
  his_pts = tf.reshape(
    his_pts, [quantity, max_input_length, keypoint_dim, 2])
  fut_pts = tf.reshape(
    fut_pts, [quantity, max_output_length, keypoint_dim, 2])
  # Run data augmentation on fut_pts. 
  return his_pts, fut_pts

def split_image_seq(source_images, his_lens, fut_lens,
                    max_input_length, max_output_length):
  shp = source_images.get_shape().as_list()
  quantity, im_height, im_width = shp[0], shp[2], shp[3]
  # Apply pyfunc.
  his_images, fut_images = tf.py_func(preprocess_data_split,
    [source_images, his_lens, fut_lens, max_input_length, max_output_length],
    [tf.float32, tf.float32], name='split_image_seq')
  his_images = tf.reshape(
    his_images, [quantity, max_input_length, im_height, im_width, 3])
  fut_images = tf.reshape(
    fut_images, [quantity, max_output_length, im_height, im_width, 3])
  return his_images, fut_images

def get_tfrecord(dataset_dir, dataset_name, split_name,
                 shuffle=True, sample_length=74):
  MAX_LEN = 600
  keypoint_dim = 32
  filenames = get_filenames(dataset_dir, dataset_name, split_name)
  input_queue = tf.train.string_input_producer(
    [os.path.join(dataset_dir, '%s/%s.tfrecords') % (dataset_name, split_name)])
  _, serialized_example = tf.TFRecordReader().read(input_queue)
  features = tf.parse_single_example(
    serialized_example,
    features={
      'seq_len': tf.FixedLenFeature([], tf.int64),
      'pts_seq': tf.FixedLenFeature(
        [MAX_LEN, keypoint_dim, 2], tf.float32),
    })
  #
  inputs = dict()
  inputs['dataset_size'] = len(filenames)
  seq_len = tf.to_int32(features['seq_len'])
  pts_seq = features['pts_seq']
  start_idx = tf.random_uniform([], minval=0, maxval=seq_len-sample_length, dtype=tf.int32)
  inputs['landmarks'] = tf.reshape(
    pts_seq[start_idx:start_idx+sample_length], [sample_length, keypoint_dim, 2])
  return inputs

def get_rawdata(dataset_dir, dataset_name, split_name,
                max_length, skip_n_frame, shuffle=True,
                common_queue_capacity=256):
  """Provides input data for a specific (dataset, split)."""
  metadata = DATASET_TO_METADATA[dataset_name]
  keypoint_dim = metadata['keypoint_dim']
  im_height = metadata['im_height']
  im_width = metadata['im_width']
  #########################
  ## Prepare Input Queue ##
  #########################
  filename_queue = get_dataset(dataset_dir, dataset_name, split_name)
  input_queue = tf.train.slice_input_producer(
    filename_queue, shuffle=shuffle, capacity=common_queue_capacity)

  inputs = dict()
  inputs['dataset_size'] = len(filename_queue[0])
  ########################
  ## Prepare Dictionary ##
  ########################
  keyframes, inputs['landmarks'] = tf.py_func(
      sample_pts_seq,
      [dataset_name, input_queue[0], max_length, skip_n_frame],
      [tf.int32, tf.float32], name='sample_pts_seq')
  keyframes = tf.reshape(keyframes, [max_length])
  inputs['images'] = tf.py_func(
    sample_image_seq,
    [dataset_name, input_queue[1], max_length, keyframes],
    tf.float32, name='sample_image_seq')
  #####################
  ## Reshape Tensors ##
  #####################
  inputs['landmarks'] = tf.reshape(
    inputs['landmarks'], [max_length, keypoint_dim, 2])
  inputs['images'] = tf.reshape(
    inputs['images'], [max_length, im_height, im_width, 3])
  return inputs

