"""Implements the video processing function."""

import StringIO
import numpy as np
from PIL import Image
import os
import glob
import subprocess

def _save_image(inp_array, image_file):
  """Function that dumps the image to disk."""
  inp_array = np.clip(inp_array, 0, 255).astype(np.uint8)
  image = Image.fromarray(inp_array)
  buf = StringIO.StringIO()
  if os.path.splitext(image_file)[1] == '.jpg':
    image.save(buf, format='JPEG')
  elif os.path.splitext(image_file)[1] == '.png':
    image.save(buf, format='PNG')
  else:
    raise ValueError('image file ends with .jpg or .png')
  with open(image_file, 'w') as f:
    f.write(buf.getvalue())

class VideoProc(object):
  """Class wrapper for video processing."""
  def __init__(self):
    pass

  def save_img_seq_to_video(self, img_seq, video_dir, video_file,
                            frame_rate=30, codec='libx264', override=False):
    """Saves the image sequence to video."""
    num_frames = img_seq.shape[0]
    num_digits = len(str(num_frames))
    video_name = os.path.splitext(video_file)[0]
    input_string = '%s_%sd.%s' \
      % (os.path.join(video_dir, video_name), '%' + '%02d' % num_digits, 'png')
    for t in xrange(num_frames):
      _save_image(img_seq[t], input_string % t)
    if override:
      ffmpeg_override = 'ffmpeg -y'
    else:
      ffmpeg_override = 'ffmpeg'
    if codec=='libx264':
      vcodec = '-vcodec %s' % codec
    elif codec=='mpeg4':
      vcodec = '-vcodec %s' % codec
    else:
      vcodec = ''
    sys_cl = '%s -f image2 -framerate %d -i %s %s -crf 25 %s -loglevel panic' \
      % (ffmpeg_override, frame_rate, input_string, vcodec, os.path.join(video_dir, video_file))
    print('Running syscall: %s' % sys_cl)
    os.system(sys_cl)
    for t in xrange(num_frames):
      os.remove(input_string % t)

  def merge_video_side_by_side(self, video_dir, source_list, video_file,
                               override=False):
    """Merge videos side by side."""
    video_dir = os.path.realpath(video_dir)
    num_videos = len(source_list)
    if override:
      ffmpeg_override = '-y'
    else:
      ffmpeg_override = '' 
    sys_cl = 'ffmpeg'
    for i in xrange(num_videos):
      sys_cl += ' -i %s' % os.path.join(video_dir, source_list[i])
    sys_cl += ' -filter_complex \"[0:v][1:v]hstack[g1];'
    for i in xrange(1, num_videos-2):
      sys_cl += ' [g%d][%d:v]hstack[g%d];' % (i, i+1, i+1)
    sys_cl += ' [g%d][%d:v]hstack\"' % (num_videos-2, num_videos-1)
    sys_cl += ' %s %s -loglevel panic' % (ffmpeg_override, os.path.join(video_dir, video_file))
    print('Running syscall: ' + sys_cl)
    os.system(sys_cl)
    for i in xrange(num_videos):
      os.remove(os.path.join(video_dir, source_list[i]))

  # TODO(xcyan): refactor this function.
  def convert_to_video_v2(self, input_dir, input_pattern, 
                          video_filename, frame_rate=30, 
                          codec='libx264', override=False):
    return self.convert_to_video(
      input_dir, os.path.join(input_dir, video_filename),
      frame_rate, codec, input_pattern, override)
                          
  def convert_to_video(self, input_dir, video_file,
                       frame_rate=30,  codec='libx264',
                       input_pattern='', override=False):
    """Converts the image sequnece to video."""
    for file_ext in ['jpg', 'png']:
      image_files = glob.glob(
        os.path.join(input_dir, '%s*.%s' % (input_pattern, file_ext)))
      if len(image_files) > 0:
        break
    if len(image_files) == 0:
      raise ValueError('Missing images (.jpg or .png) in the folder [%s].' \
        % input_dir)
    last_us_pos = image_files[0].rfind('_')
    last_dot_pos = image_files[0].rfind('.')
    num_digits = last_dot_pos - last_us_pos - 1
    if override:
      ffmpeg_override = 'ffmpeg -y'
    else:
      ffmpeg_override = 'ffmpeg'
    if codec=='libx264':
      vcodec = '-vcodec %s' % codec
    else:
      vcodec = ''
    input_string = '%s_%sd.%s' \
      % (image_files[0][:last_us_pos], '%' + '%02d' % num_digits, file_ext)
    sys_cl = '%s -f image2 -framerate %d -i %s %s -crf 25 %s -loglevel panic' \
      % (ffmpeg_override, frame_rate, input_string, vcodec, video_file)
    print('Running syscall: %s' % sys_cl)
    os.system(sys_cl)

  def convert_dataset_to_video(self, dataset_dir, video_dir,
                               file_prefix='vid'):
    """Converts the entire image dataset to video dataset."""
    f_list = glob.glob('%s/*' % dataset_dir)
    has_images = False
    for sub_dir in f_list:
      slash_pos = sub_dir.rfind('/')
      f = sub_dir[slash_pos+1:]
      if os.path.isdir(sub_dir):
        self.convert_dataset_to_video(sub_dir, video_dir, file_prefix + '_%s' % f)
      elif os.path.splitext(f)[1] in ['.jpg', '.png']:
        has_images = True
    if has_images:
      self.convert_to_video(dataset_dir,
                            os.path.join(video_dir, file_prefix + '.mp4'))
 

