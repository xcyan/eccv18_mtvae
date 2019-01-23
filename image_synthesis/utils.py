#import imageio
import random
import math
import cv2
import time
import numpy as np
import scipy.io as sio
from os import listdir, makedirs, system

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

def transform(X):
  return X/127.5 - 1

def inverse_transform(X):
  return (X+1.)/2.

def get_minibatches_idx(n, minibatch_size, shuffle=False):
  """
  Used to shuffle the dataset at each iteration.
  """

  idx_list = np.arange(n, dtype="int32")

  if shuffle:
      random.shuffle(idx_list)

  minibatches = []
  minibatch_start = 0
  for i in range(n // minibatch_size):
      minibatches.append(idx_list[minibatch_start:
                                  minibatch_start + minibatch_size])
      minibatch_start += minibatch_size

  if (minibatch_start != n):
      # Make a minibatch out of what is left
      minibatches.append(idx_list[minibatch_start:])

  return zip(range(len(minibatches)), minibatches)

def gauss2D_mask(center, shape, sigma=0.5):
  m,n = [ss-1 for ss in shape]
  y,x = np.ogrid[0:m+1,0:n+1]
  y = y - center[0]
  x = x - center[1]
  z = x*x + y*y
  h = np.exp( -z / (2.*sigma*sigma) )
  sumh = h.sum()
  if sumh != 0:
    h = h/sumh
  return h

def get_tube(posey, posex, img, pad=10):
  box       = np.array([posex.min(),
                        posey.min(),
                        posex.max(),
                        posey.max()])
  x1 = box[0]-pad
  x2 = box[2]+pad
  y1 = box[1]-pad
  y2 = box[3]+pad

  h = y2-y1+1
  w = x2-x1+1
  if h > w:
    left_pad  = (h-w)/2
    right_pad = (h-w)/2 + (h-w)%2
    x1 = x1 - left_pad
    if x1 < 0: x1 = 0
    x2 = x2 + right_pad
    if x2 > img.shape[1]: x2 = img.shape[1]
  elif w > h:
    up_pad    = (w-h)/2
    down_pad  = (w-h)/2 + (w-h)%2
    y1 = y1 - up_pad
    if y1 < 0: y1 = 0
    y2 = y2 + down_pad
    if y2 > img.shape[0]: y2 = img.shape[0]

  if y1 >= 0:
    cposey=posey-y1
  else:
    cposey=posey-box[1]

  if x1 >= 0:
    cposex=posex-x1
  else:
    cposex=posex-box[0]

  if y1 < 0: y1 = 0
  if x1 < 0: x1 = 0

  box = np.array([x1,y1,x2,y2])
  return cposey.astype('int32'), cposex.astype('int32'), box.astype('int32')

def load_penn_data_bl(f_name, data_path, image_size, steps):
  flip      = np.random.binomial(1,.5,1)[0]
  vid_path  = f_name.split()[0]
  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  low       = 0
  high      = len(vid_imgs)-steps+1
  if high < 1:
    high = 1
  stidx = np.random.randint(low=0, high=high)
  seq   = np.zeros((image_size, image_size, steps, 3), dtype='float32')
  for t in range(steps):
    img   = cv2.imread(vid_path+'/'+vid_imgs[np.min([stidx+t,len(vid_imgs)-1])])
    tks   = vid_path.split('frames')
    ff    = sio.loadmat(tks[0]+'labels'+'/ruben_'+tks[1][1:]+'.mat')
    img = cv2.resize(img,(image_size,image_size))
    seq[:,:,t] = transform(img)

  if flip == 1:
    seq  = seq[:,::-1]

  return seq


def load_h36m_data_bl(f_name, image_size, steps):
  flip     = np.random.binomial(1,.5,1)[0]
  vid_path = f_name.split('\n')[0].split('.mp4')[0]
  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  fskip    = 4

  nframes   = len(vid_imgs)
  high      = nframes-fskip*steps+1
  stidx     = np.random.randint(low=0, high=high)

  seq   = np.zeros((image_size, image_size, steps, 3), dtype='float32')
  for t in range(steps):
    img   = cv2.imread(vid_path+'/'+vid_imgs[stidx+t*fskip])
    img   = cv2.resize(img,(image_size,image_size))
    seq[:,:,t] = transform(img)

  if flip == 1:
    seq  = seq[:,::-1]

  return seq


def load_penn_data(f_name, data_path, image_size, steps):
  lines = [[0,0,1,2],
           [1,1,2,2],
           [1,1,3,3],
           [3,3,5,5],
           [2,2,4,4],
           [4,4,6,6],
           [1,2,7,8],
           [7,7,8,8],
           [7,7,9,9],
           [9,9,11,11],
           [8,8,10,10],
           [10,10,12,12]]

  rnd_steps = np.random.randint(1,steps)
  flip      = np.random.binomial(1,.5,1)[0]
  vid_path  = f_name.split()[0]
  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
  low       = 0
  high      = len(vid_imgs)-rnd_steps-1
  if high < 1:
    rnd_steps = rnd_steps + high
    high = 1
  stidx = np.random.randint(low=0, high=high)
  seq   = np.zeros((image_size, image_size, 2, 3), dtype='float32')
  pose  = np.zeros((image_size, image_size, 2, 48), dtype='float32')
  for t in range(2):
    img   = cv2.imread(vid_path+'/'+vid_imgs[stidx+t*rnd_steps])
    cpose = np.zeros((img.shape[0], img.shape[1], 48))
    tks   = vid_path.split('frames')
    ff    = sio.loadmat(tks[0]+'labels'+'/ruben_'+tks[1][1:]+'.mat')
    posey = ff['y'][stidx+t*rnd_steps]
    posex = ff['x'][stidx+t*rnd_steps]
    visib = ff['visibility'][stidx+t*rnd_steps]
    for j in range(12):
      if visib[lines[j][0]] and visib[lines[j][1]] and \
         visib[lines[j][2]] and visib[lines[j][3]]:
        interp_x = np.linspace((posex[lines[j][0]]+posex[lines[j][1]])/2,
                               (posex[lines[j][2]]+posex[lines[j][3]])/2,
                               4, True)
        interp_y = np.linspace((posey[lines[j][0]]+posey[lines[j][1]])/2,
                               (posey[lines[j][2]]+posey[lines[j][3]])/2,
                               4, True)
        for k in range(4):
            gmask = gauss2D_mask((interp_y[k], interp_x[k]),
                                  img.shape[:2], sigma=8.)
            cpose[:,:,j*4+k] = gmask/gmask.max()
      else:
        if visib[lines[j][0]] and visib[lines[j][1]]:
          point_x = (posex[lines[j][0]]+posex[lines[j][1]])/2
          point_y = (posey[lines[j][0]]+posey[lines[j][1]])/2
          gmask   = gauss2D_mask((point_y, point_x),img.shape[:2], sigma=8.)
          cpose[:,:,j*4] = gmask/gmask.max()
        if visib[lines[j][2]] and visib[lines[j][3]]:
          point_x = (posex[lines[j][2]]+posex[lines[j][3]])/2
          point_y = (posey[lines[j][2]]+posey[lines[j][3]])/2
          gmask   = gauss2D_mask((point_y, point_x),img.shape[:2], sigma=8.)
          cpose[:,:,(j+1)*4-1] = gmask/gmask.max()

    img = cv2.resize(img,(image_size,image_size))
    cpose = cv2.resize(cpose,(image_size,image_size))
    seq[:,:,t] = transform(img)
    pose[:,:,t] = cpose

  if flip == 1:
    seq  = seq[:,::-1]
    pose = pose[:,::-1]

  return seq, pose


def load_h36m_data(f_name, image_size, steps):
  rnd_steps = np.random.randint(1,steps)
  flip      = np.random.binomial(1,.5,1)[0]
  vid_path  = f_name.split('\n')[0].split('.mp4')[0]

  vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])

  anno_path = vid_path.split('Videos')[0]+'MyPoseFeatures/D2_Positions'\
            + vid_path.split('Videos')[1]+' posentube.npz'

  pose_data = np.load(anno_path)
  all_posey = pose_data['all_posey']
  all_posex = pose_data['all_posex']
  box = pose_data['box']

  high = np.min([all_posey.shape[0],len(vid_imgs)])-rnd_steps-1
  if high < 1:
    rnd_steps = rnd_steps + high
    high = 1

  stidx = np.random.randint(low=0, high=high)

  n_joints = all_posex.shape[1]
  seq = np.zeros((image_size, image_size, 2, 3), dtype='float32')
  pose = np.zeros((image_size, image_size, 2, n_joints), dtype='float32')
  shape = [box[2]-box[0],box[3]-box[1]]
  for t in range(2):
    posey = all_posey[stidx+t*rnd_steps,:]
    posex = all_posex[stidx+t*rnd_steps,:]
    img = cv2.imread(vid_path+'/'+vid_imgs[stidx+t*rnd_steps])
    cpose = np.zeros((shape[0],shape[1], n_joints), dtype='float32')
    for j in range(n_joints):
      gmask = gauss2D_mask((posey[j], posex[j]), shape, sigma=8.)
      cpose[:,:,j] = gmask/gmask.max()
    img = cv2.resize(img,(image_size,image_size))
    cpose = cv2.resize(cpose,(image_size,image_size))
    seq[:,:,t] = transform(img)
    pose[:,:,t] = cpose

  return seq, pose


_COLOR_TO_VALUE = {
  'red': [0, 0, 255],
  'green': [0, 255, 0],
  'white': [255, 255, 255],
  'blue': [255, 0, 0],
  'yellow': [0, 255, 255],
  'purple': [128, 0, 128],
}


def visualize_boundary(inp_array, radius=2, colormap='green'):
  im_height = inp_array.shape[0]
  im_width = inp_array.shape[1]
  colorvalue = _COLOR_TO_VALUE[colormap]
  for i in range(im_height):
    for ch in range(3):
      inp_array[i, 0:radius, ch] = colorvalue[ch]
      inp_array[i, -radius:, ch] = colorvalue[ch]
  for i in range(im_width):
    for ch in range(3):
      inp_array[0:radius, i, ch] = colorvalue[ch]
      inp_array[-radius:, i, ch] = colorvalue[ch]
  inp_array = inp_array.astype(np.uint8)
  return inp_array


def draw_frame(img, is_input):
  if is_input:
    visualize_boundary(img, radius=4, colormap='green')
  else:
    visualize_boundary(img, radius=4, colormap='red')
  return img


def draw_frameAB(img, is_input):
  if is_input:
    visualize_boundary(img, radius=4, colormap='green')
  else:
    visualize_boundary(img, radius=4, colormap='blue')

  return img


def draw_frameCD(img, is_input):
  if is_input:
    visualize_boundary(img, radius=4, colormap='purple')
  else:
    visualize_boundary(img, radius=4, colormap='red')
  return img

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

def visualize_h36m_skeleton(inp_array, landmarks, radius=2): 
  for indx, p_pair in enumerate(SKELETON_MAP):
    p_A = p_pair[0]
    p_B = p_pair[1]
    colormap = SKELETON_COLOR[indx]
    inp_array = draw_line(inp_array, landmarks[p_A], landmarks[p_B], 
                          radius=radius, colormap=colormap)
  return inp_array

def visualize_lm(posex, posey, w, h, lm_size):
  pose_image = np.zeros((h, w, 3))
  landmarks = np.stack((posex, posey), axis=1)
  pose_image = visualize_h36m_skeleton(pose_image, landmarks)

  cpose = np.zeros((h, w, lm_size))
  for j in range(lm_size):
    gmask   = gauss2D_mask((posey[j], posex[j]),(h,w), sigma=8.)
    cpose[:,:,j] = gmask/gmask.max()
  return pose_image, cpose



