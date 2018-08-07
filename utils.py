"""Utility functions."""

try: 
  from StringIO import StringIO
except:
  from cStringIO import StringIO

import numpy as np
from PIL import Image
from scipy.io import loadmat
import scipy
import csv
import os
import glob
import random
import time
import math
from preprocess.ffmpeg_reader import FFMPEG_VideoReader
import pickle

TARGET_FPS = 30

def save_python_objects(objs, filename):
  with open(filename, 'wb') as f:
    pickle.dump(objs, f)

def load_python_objects(filename):
  with open(filename, 'rb') as f:
    objs = pickle.load(f)
  return objs

# TODO(xcyan): install cv2 for speed up.
def load_image(image_file, out_size=None):
  inp_array = Image.open(image_file)
  if out_size is not None:
    inp_array = inp_array.resize(out_size)
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  return inp_array
  
def save_image(inp_array, image_file):
  """Function that dumps the image to disk."""
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  image = Image.fromarray(inp_array)
  buf = StringIO()
  if os.path.splitext(image_file)[1] == '.jpg':
    image.save(buf, format='JPEG')
  elif os.path.splitext(image_file)[1] == '.png':
    image.save(buf, format='PNG')
  else:
    raise ValueError('image file ends with .jpg or .png')
  with open(image_file, 'w') as f:
    f.write(buf.getvalue())

# Read fitting
def read_single_fitting(filename):
  """Python wrapper that loads fitting parameters from .mat datafile."""
  coeff_struct = loadmat(filename)
  id_coeff = np.asarray(coeff_struct['normalized_shape_coeff'], dtype=np.float32)
  expr_coeff = np.asarray(coeff_struct['normalized_exp_coeff'], dtype=np.float32)
  pose_para = np.asarray(coeff_struct['Pose_Para'], dtype=np.float32)
  # Remove the t3(z) from the pose parameters.
  pose_para = pose_para[[0, 1, 2, 3, 4, 6]] 
  return id_coeff, expr_coeff, pose_para

def read_seq_fitting(filename, num_frames, skip_frames):
  """Python wrapper that loads fitting parameter sequence from .mat datafile."""
  coeff_struct = loadmat(filename)
  id_coeff = np.asarray(coeff_struct['seq_shape_coeff'], dtype=np.float32)
  # TODO(xcyan): remove this hacky implementation.
  video_length = id_coeff.shape[0] - 1
  keyframes = np.zeros((num_frames), dtype=np.int32)
  start_frame = random.randint(
    0, video_length-num_frames*skip_frames)
  for t in xrange(num_frames):
    keyframes[t] = int(start_frame + t * skip_frames)
  id_coeff = id_coeff[keyframes, :]
  expr_coeff = np.asarray(coeff_struct['seq_exp_coeff'], dtype=np.float32)
  expr_coeff = expr_coeff[keyframes, :]
  pose_para = np.asarray(coeff_struct['seq_pose_para'], dtype=np.float32)
  pose_para = pose_para[keyframes, :]
  pose_para = pose_para[:, [0, 1, 2, 3, 4, 6]]
  return id_coeff, expr_coeff, pose_para, keyframes

# Read pts (landmarks)
def read_single_pts(filename, keypoint_dim):
  """Python wrapper that loads landmark from .csv datafile."""
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    pts = [row for row in csvreader]
  pts = np.asarray(pts, dtype=np.float32)
  return pts

def read_seq_pts(filename, keyframes, keypoint_dim):
  """Python wrapper that loads landmark sequence from .csv datafile."""
  num_frames = keyframes.shape[0]
  pts = np.zeros((num_frames, keypoint_dim, 2), dtype=np.float32)
  with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    row_index = 0
    for row in csvreader:
      if len(row) == 2:
        continue
      x_coord = np.asarray(row, dtype=np.float32)[keyframes * 2]
      y_coord = np.asarray(row, dtype=np.float32)[keyframes * 2 + 1]
      x_coord = np.reshape(x_coord, (1, 1, num_frames))
      y_coord = np.reshape(y_coord, (1, 1, num_frames))
      pts[:, row_index, 0] = x_coord
      pts[:, row_index, 1] = y_coord
      row_index += 1
  return pts

# Read emotion score vector.
def read_seq_emo(annot_emo, video_length, num_frames):
  """Python wrapper that loads emotion score sequence from .csv datafile."""
  #seq_emo = np.zeros((video_length, 7), dtype=np.float32)
  with open(annot_emo, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    seq_emo = [row for row in csvreader]
  seq_emo = np.asarray(seq_emo, dtype=np.float32)
  if seq_emo.shape[0] < video_length:
    video_length = np.copy(seq_emo.shape[0])
  start_pos = 0
  end_pos = video_length - 1
  for t in xrange(int(num_frames / 2), video_length - num_frames):
    # When the neutral face score is below 60.0%.
    if seq_emo[t, 0] > 0 and seq_emo[t, 0] <= 50.0:
      start_pos = t - int(num_frames / 2)
      end_pos = start_pos + num_frames
      break
  #for t in xrange(start_pos + num_frames, video_length):
  # if seq_emo[t, 0] > 0 and seq_emo[t, 0] > 60.0:
  #   end_pos = t
  #   break
  return start_pos, end_pos

def read_video(video_file, keyframes, random_chunk=True):
  ts = time.time()
  video_reader = FFMPEG_VideoReader(video_file, target_fps=TARGET_FPS)
  _class_init_time = time.time() - ts

  ts = time.time()
  frame_width, frame_height = video_reader.size
  video_length = video_reader.nframes
  fps = video_reader.fps
  num_frames = keyframes.shape[0]
  if random_chunk:
    init_frame = video_reader.get_frame(1. * keyframes[0] / fps)[:, :, :3]
  _parse_info_time = time.time() - ts
  
  ts = time.time()
  output_video = np.zeros(
    (num_frames, frame_height, frame_width, 3), dtype=np.uint8)
  output_video[0] = init_frame
  for t in xrange(1, num_frames):
    output_video[t] = video_reader.read_frame()[:, :, :3]
    if t == num_frames - 1:
      break
    video_reader.skip_frames(keyframes[t]-keyframes[t-1]-1)
  video_reader.close()
  _read_frame_time = time.time() - ts
  print('Reading video: class init [%.4f s], parse info [%.4f s], read_frames [%.4f s]' \
    % (_class_init_time, _parse_info_time, _read_frame_time))
  return output_video

# Read video_v2
def read_video_v2(video_file, annot_emo, num_frames, random_chunk=True):
  """Loads video and segments videom based on emotion score."""
  if isinstance(video_file, bytes):
    video_file = video_file.decode('utf-8')
  if isinstance(num_frames, np.int32):
    num_frames = int(num_frames)
  
  ts = time.time()
  video_reader = FFMPEG_VideoReader(video_file, target_fps=TARGET_FPS)
  _class_init_time = time.time() - ts

  ts = time.time()
  frame_width, frame_height = video_reader.size
  video_length = video_reader.nframes
  fps = video_reader.fps
  sample_rate = 1
  if annot_emo is None:
    start_pos = 0
    end_pos = video_length - 1
  else:
    # TODO(xcyan): remove the redundant 'num_frames'.
    start_pos, end_pos = read_seq_emo(annot_emo, video_length, num_frames)
  if random_chunk:
    sample_rate = int((end_pos - start_pos + 1) / num_frames)
    expected_end_pos = start_pos + sample_rate * (num_frames - 1)
    if expected_end_pos < end_pos:
      start_pos += random.randint(0, end_pos - expected_end_pos)
    video_reader.get_frame(1. * start_pos / fps)
  _parse_info_time = time.time() - ts

  ts = time.time()
  output_video = np.zeros(
    (num_frames, frame_height, frame_width, 3), dtype=np.uint8)
  keyframes = np.zeros((num_frames), dtype=np.int32)
  for frame_id in xrange(num_frames):
    output_video[frame_id] = video_reader.read_frame()[:, :, :3]
    keyframes[frame_id] = start_pos + frame_id * sample_rate
    if frame_id == num_frames - 1:
      break
    if sample_rate > 1:
      video_reader.skip_frames(sample_rate - 1)
  video_reader.close()
  _read_frame_time = time.time() - ts
  print('Reading video: class init [%.4f s], parse info [%d-%d], [%.4f s], read_frames [%.4f s]' \
    % (_class_init_time, start_pos, end_pos, _parse_info_time, _read_frame_time))
  return output_video, keyframes  

# 
def convert_to_img_seq(inp_video):
  video_reader = FFMPEG_VideoReader(inp_video, target_fps=30.0)
  # TODO(xcyan): remove the hacks here (ffmpeg and imageio are incompatible).
  num_frames = video_reader.nframes-1
  im_height, im_width = video_reader.size
  video_reader.get_frame(0.0)
  img_seq = []
  metadata = dict()
  for t in xrange(num_frames):
    img_seq.append(video_reader.read_frame()[:, :, :3])
  metadata['video_length'] = num_frames
  metadata['frame_height'] = im_height
  metadata['frame_width'] = im_width
  return img_seq, metadata
  
def rescale_img_seq(img_seq, img_size):
  num_frames = len(img_seq)
  im_height, im_width = img_seq[0].shape[0], img_seq[0].shape[1]
  face_im_seq = np.zeros(
    (num_frames, img_size, img_size, 3),
    dtype=np.float32)
  for t in xrange(num_frames):
    face_im_seq[t] = scipy.misc.imresize(
      img_seq[t], (img_size, img_size))
  return face_im_seq

def parse_video_path(inp_video):
  slash_pos = inp_video.rfind('/')
  video_name = os.path.splitext(inp_video[slash_pos+1:])[0]
  root_path = inp_video[:slash_pos]
  root_path = root_path[:root_path.rfind('/')]
  return root_path, video_name

def force_mkdir(dir_name):
  if not os.path.isdir(dir_name):
    os.mkdir(dir_name)

def force_rmfile(filename):
  if os.path.exists(filename):
    os.remove(filename)

_COLOR_TO_VALUE = {
  'red': [255, 0, 0],
  'green': [0, 255, 0],
  'white': [255, 255, 255],
  'blue': [0, 0, 255],
  'yellow': [255, 255, 0],
  'purple': [128, 0, 128],
}

def visualize_boundary(inp_array, radius=2, colormap='green'):
  im_height = inp_array.shape[0]
  im_width = inp_array.shape[1]
  colorvalue = _COLOR_TO_VALUE[colormap]
  for i in xrange(im_height):
    for ch in xrange(3):
      inp_array[i, 0:radius, ch] = colorvalue[ch]
      inp_array[i, -radius:, ch] = colorvalue[ch]
  for i in xrange(im_width):
    for ch in xrange(3):
      inp_array[0:radius, i, ch] = colorvalue[ch]
      inp_array[-radius:, i, ch] = colorvalue[ch]
  inp_array = inp_array.astype(np.uint8)
  return inp_array

def visualize_normals_batch_seq(inp_array, radius=2, seq_in_size=1):
  batch_size = inp_array.shape[0]
  seq_size = inp_array.shape[1]
  inp_array = inp_array.astype(np.uint8)
  for i in xrange(batch_size):
    for t in xrange(seq_size):
      if t < seq_in_size:
        colormap = 'green'
      else:
        colormap = 'red' 
      inp_array[i, t] = visualize_boundary(
        inp_array[i, t], radius, colormap)
  return inp_array

def visualize_landmarks_batch_seq(inp_array, landmarks,
                                  radius=2, seq_in_size=1):
  batch_size = inp_array.shape[0]
  seq_size = inp_array.shape[1]
  inp_array = inp_array.astype(np.uint8)
  for i in xrange(batch_size):
    for t in xrange(seq_size):
      inp_array[i, t] = visualize_landmarks(
        inp_array[i, t], landmarks[i, t], radius=radius)
      if t < seq_in_size:
        colormap = 'green'
      else:
        colormap = 'red'
      # else:
      #  colormap = 'white' 
      inp_array[i, t] = visualize_boundary(
        inp_array[i, t], radius, colormap)
  return inp_array

def draw_line(inp_array, pointA, pointB, radius=2, colormap='white'):
  im_height = inp_array.shape[0]
  im_width = inp_array.shape[1]
  colorvalue = _COLOR_TO_VALUE[colormap]
  if abs(pointA[0] - pointB[0]) < 1 and abs(pointA[1] - pointB[1]) < 1:
    return inp_array
  if abs(pointA[0] - pointB[0]) < 1:
    hmin = int(math.ceil(min(pointA[1], pointB[1])))
    hmax = int(math.floor(max(pointA[1], pointB[1])))
    est_w = (pointA[0] + pointB[0]) / 2
    wmin = int(est_w - radius)
    wmax = int(est_w + radius)  
    for h_orig in xrange(hmin, hmax+1):
      for w_orig in xrange(wmin, wmax+1):
        h = np.clip(h_orig, 0, im_height-1)
        w = np.clip(w_orig, 0, im_width-1)
        for ch in xrange(3):
          inp_array[h, w, ch] = colorvalue[ch] 
    return inp_array
  if abs(pointA[1] - pointB[1]) < 1:
    wmin = int(math.ceil(min(pointA[0], pointB[0])))
    wmax = int(math.floor(max(pointA[0], pointB[0])))
    est_h = (pointA[1] + pointB[1]) / 2
    hmin = int(est_h - radius)
    hmax = int(est_h + radius)
    for h_orig in xrange(hmin, hmax+1):
      for w_orig in xrange(wmin, wmax+1):
        h = np.clip(h_orig, 0, im_height-1)
        w = np.clip(w_orig, 0, im_width-1)
        for ch in xrange(3):
          inp_array[h, w, ch] = colorvalue[ch]
    return inp_array
  ###########
  ## Final ##
  ###########
  hmin = int(math.ceil(min(pointA[1], pointB[1])))
  hmax = int(math.floor(max(pointA[1], pointB[1])))
  for h_orig in xrange(hmin, hmax+1):
    est_w = ((h_orig - pointA[1]) * pointB[0] + (pointB[1] - h_orig) * pointA[0])/(pointB[1] - pointA[1])
    wmin = int(est_w - radius)
    wmax = int(est_w + radius)
    for w_orig in xrange(wmin, wmax+1):
      h = np.clip(h_orig, 0, im_height-1)
      w = np.clip(w_orig, 0, im_width-1)
      for ch in xrange(3):
        inp_array[h, w, ch] = colorvalue[ch]
  return inp_array  

def visualize_landmarks(inp_array, landmarks, radius=2):
  im_height = inp_array.shape[0]
  im_width = inp_array.shape[1]
  num_parts = landmarks.shape[0]
  for i in xrange(num_parts):
    if landmarks[i, 0] < 0 or landmarks[i, 0] >= im_width: 
      continue 
    if landmarks[i, 1] < 0 or landmarks[i, 1] >= im_height:
      continue
    wmin = int(landmarks[i, 0] - radius)
    wmax = int(landmarks[i, 0] + radius)
    hmin = int(landmarks[i, 1] - radius)
    hmax = int(landmarks[i, 1] + radius)
    for h_orig in xrange(hmin, hmax):
      for w_orig in xrange(wmin, wmax):
        h = np.clip(h_orig, 0, im_height-1)
        w = np.clip(w_orig, 0, im_width-1)
        inp_array[h, w, 0] = 0
        inp_array[h, w, 1] = 0
        inp_array[h, w, 2] = 255
  inp_array = inp_array.astype(np.uint8)
  return inp_array

SKELETON_MAP = [[1, 2], [2, 3],
                [6, 7], [7, 8],
                [6, 11], [1, 11],
                [11, 12], [12, 13], [13, 14], [14, 15],
                [24, 25], [25, 26], [26, 27],
                [16, 17], [17, 18], [18, 19]]
SKELETON_COLOR = ['red', 'red', 'green', 'green', 'yellow', 'yellow', 
                  'yellow', 'yellow', 'yellow', 'yellow',
                  'red', 'red', 'red',
                  'green', 'green', 'green']

def visualize_h36m_skeleton(inp_array, landmarks, radius=2):
  #inp_array = visualize_landmarks(inp_array, landmarks, radius=2)
  for indx, p_pair in enumerate(SKELETON_MAP):
    p_A = p_pair[0]
    p_B = p_pair[1]
    colormap = SKELETON_COLOR[indx]
    inp_array = draw_line(inp_array, landmarks[p_A], landmarks[p_B], 
                          radius=radius, colormap=colormap)
  return inp_array

def visualize_h36m_skeleton_batch(landmark_seq, img_size, radius=2):
  batch_size = landmark_seq.shape[0]
  inp_array = np.zeros((batch_size, img_size, img_size, 3), dtype=np.float32)
  for i in xrange(batch_size):
    rescaled_lms = np.copy(landmark_seq[i]) * img_size
    inp_array[i] = visualize_h36m_skeleton(inp_array[i], rescaled_lms, radius)
  return inp_array

def flatten_img_seq(inp_array):
  quantity, im_height, im_width = inp_array.shape[0], inp_array.shape[1], inp_array.shape[2]
  out_img = np.zeros((im_height, quantity * im_width, 3), dtype=np.float32)
  for t in xrange(quantity):
    out_img[:, t*im_width:(t+1)*im_width] = np.copy(inp_array[t])
  return out_img        

