import os
import cv2
import sys
import time
# import ssim
# import imageio
import pickle

import tensorflow as tf
#import scipy.misc as sm
#import scipy.io as sio
import numpy as np
# import skimage.measure as measure

from posenet import POSENET
from utils import *
#from utils import *
from os import listdir, makedirs, system
from os.path import exists
#from argparse import ArgumentParser
# from joblib import Parallel, delayed
#from pylab import *
#from skimage.draw import line_aa
# from PIL import Image
# from PIL import ImageDraw

HIS_KEYFRAMES = [0, 7, 15]
FUT_KEYFRAMES = [0, 3, 15, 31, 63]
LM_SIZE = 32

class ImageAnalogyNet(object):
  def __init__(self, seen_step, fut_step, getgt=False):

    checkpoint_dir = 'checkpoints/posenet_model/'
    self.fskip = 1
    self.image_size = 128
    self.seen_step = seen_step
    self.fut_step = fut_step
    if not getgt:
      self.model = POSENET(image_size=self.image_size, batch_size=1,
                           is_train=False, n_joints=32)
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
      self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                   log_device_placement=False,
                                                   gpu_options=gpu_options))
  
      self.sess.run(tf.global_variables_initializer())
      if self.model.load(self.sess, checkpoint_dir):
        print(" [*] Load SUCCESS")
      else:
        print(" [!] Load failed... exitting")
        sys.exit(0)
    else:
      print(" [!] Model not loaded. Only saving groundtruth frames.")
      self.model = None

  def transform(self, X):
    return X/127.5 - 1

  def inverse_transform(self, X):
    return (X+1.)/2.*255.

  def flatten_img_seq(self, inp_array):
    quantity, im_height, im_width = inp_array.shape[0], inp_array.shape[1], inp_array.shape[2]
    out_img = np.zeros((im_height, quantity * im_width, 3), dtype=np.float32)
    for t in xrange(quantity):
      out_img[:, t*im_width:(t+1)*im_width] = np.copy(inp_array[t])
    return out_img

  def run_inference(self, pose_path, model_name, pred_data, save_path):
    gt_path = 'workspace/Human3.6M/'
    data_path   = 'workspace/Human3.6M/'
    f = open(data_path+'test_list_pose.txt','r')
    temptestfiles = f.readlines()
    class_dict =  {'walkdog': 0,
                   'purchases': 1,
                   'waiting': 2,
                   'eating': 3,
                   'sitting': 4,
                   'photo': 5,
                   'discussion': 6,
                   'greeting': 7,
                   'walking': 8,
                   'phoning': 9,
                   'posing': 10,
                   'walktogether': 11,
                   'directions': 12,
                   'smoking': 13,
                   'sittingdown': 14}

    testfiles = []
    for f in temptestfiles:
      class_name = f.split('/')[-1].split()[0].split('.')[0].lower()
      if class_name in class_dict.keys():
        testfiles.append(f)

    pkl_file = open(gt_path+'alldata.pkl', 'rb')
    gt_data = pickle.load(pkl_file)
    pkl_file.close()

    for i in xrange(len(pred_data['video_id'])):
      print ' Video '+str(i)+'/'+str(len(pred_data['video_id']))
      tst_idx = pred_data['video_id'][i] - 600
      vid_path = (
          testfiles[tst_idx].split('MyPoseFeatures')[0]+
          'Videos/'+ testfiles[tst_idx].split('/')[-1].split(' pose')[0]
      )
      if not exists(vid_path): continue
      vid_imgs = sorted([f for f in listdir(vid_path) if f.endswith('.png')])
      anno_path = (vid_path.split('Videos')[0]+'MyPoseFeatures/D2_Positions'+
                   vid_path.split('Videos')[1]+' posentube.npz')
      act = testfiles[i].split('/')[-1].split()[0].split('.')[0].lower()

      samples = []
      for s in xrange(len(pred_data['pred_keypoints'][i])):
        pose_data = pred_data['pred_keypoints'][i][s].copy()
        gtdata = np.stack(
            [gt_data[pred_data['video_id'][i]]['all_posex'],
             gt_data[pred_data['video_id'][i]]['all_posey']],
            axis=-1
        )
        box = np.load(anno_path)['box']

        all_posey = ((pose_data[:,:,1]+1)/2.0*(box[3]-box[1]))[None].round()
        all_posex = ((pose_data[:,:,0]+1)/2.0*(box[2]-box[0]))[None].round()
        gtdata[:,:,1] = ((gtdata[:,:,1]+1)/2.0*(box[3]-box[1]))[None].round()
        gtdata[:,:,0] = ((gtdata[:,:,0]+1)/2.0*(box[2]-box[0]))[None].round()

        path_tks = vid_path.split('/')
        name_tks = path_tks[-1].split(' ')
        if len(name_tks) == 2:
          folder_name = path_tks[-3]+'_'+name_tks[0]+'_'+name_tks[1]
        else:
          folder_name = path_tks[-3]+'_'+name_tks[0]

        savedir = save_path + '/vid_{0:04d}_'.format(i) + folder_name
        if not os.path.exists(savedir):
          os.makedirs(savedir)

        seq_batch = np.zeros((1, self.image_size, self.image_size,
                              self.seen_step+self.fut_step, 3), dtype='float32')
        pose_batch = np.zeros((1, self.image_size, self.image_size,
                               self.seen_step+self.fut_step, 32), dtype='float32')
        bshape = [box[2]-box[0],box[3]-box[1]]
        stidx = pred_data['ob_start'][i]
        endidx = stidx+self.fskip*(self.seen_step+self.fut_step)
        cvid_imgs = vid_imgs[stidx:endidx:self.fskip]
        gtdata = gtdata[stidx:endidx:self.fskip][None].copy()

        all_posey = np.concatenate(
            (gtdata[:, :self.seen_step, :, 1], all_posey), axis=1
        )
        all_posex = np.concatenate(
            (gtdata[:, :self.seen_step, :, 0], all_posex), axis=1
        )
        for t in xrange(len(cvid_imgs)):
          posey = all_posey[0,t,:]
          posex = all_posex[0,t,:]
          img = cv2.imread(vid_path+'/'+cvid_imgs[t])
          # TODO(xcyan): xinchen added lm_size.
          cpose = visualize_lm(posex, posey, bshape[0], bshape[1], LM_SIZE)
          img = cv2.resize(img, (self.image_size, self.image_size))
          cpose = cv2.resize(cpose, (self.image_size, self.image_size))
          seq_batch[0,:,:,t]  = self.transform(img)
          pose_batch[0,:,:,t] = cpose

        t_data = seq_batch.copy()
        p_data = np.zeros(t_data.shape, dtype='float32')
        xt = seq_batch[:,:,:,self.seen_step-1]
        pt = pose_batch[:,:,:,self.seen_step-1]
        if self.model is not None:
          for t in xrange(self.fut_step):
            ptpn = pose_batch[:,:,:,self.seen_step+t]
            feed_dict = {self.model.xt_: xt,
                         self.model.pt_: pt, self.model.ptpn_: ptpn}
            pred = self.sess.run(self.model.G, feed_dict=feed_dict)
            p_data[:,:,:,t] = pred[:,:,:,0].copy()

        samples.append(
            np.concatenate((seq_batch[:,:,:,:self.seen_step], p_data), axis=3)
        )


      for t in xrange(self.seen_step+self.fut_step):
        pred = self.inverse_transform(samples[0][0,:,:,t]).astype('uint8')
        pred = draw_frame(cv2.resize(pred, (180, 180)), t<self.seen_step)
        for s in xrange(1, len(samples)):
          cpred = self.inverse_transform(samples[s][0,:,:,t]).astype('uint8')
          cpred = draw_frame(cv2.resize(cpred, (180, 180)), t<self.seen_step)
          pred = np.concatenate([pred, cpred], axis=1)
          

        if self.model is not None:
          cv2.imwrite(savedir+'/os_'+str(pred_data['ob_start'][i])+
                      '_ours_'+'{0:04d}'.format(t)+'.png', pred)


      if self.model is not None:
        #cmd1 = 'rm '+savedir+'/os_'+str(pred_data['ob_start'][i])+'_ours.gif'
        img_gif_path = save_path+'/'+'{0:02d}'.format(i)+'_merged_'+model_name+'.gif'
        pose_gif_path = pose_path+'/'+'{0:02d}'.format(i)+'_merged_'+model_name+'.gif'
        comb_gif_path = save_path+'/'+'{0:02d}'.format(i)+'_comb_'+model_name+'.gif'
        #
        cmd1 = ('ffmpeg -y -f image2 -framerate 7 -i '+savedir+
                '/os_'+str(pred_data['ob_start'][i])+ '_ours_%04d.png '+
                '-crf 25 '+
                img_gif_path)
        cmd2 = ('rm '+savedir+'/os_'+str(pred_data['ob_start'][i])+
                '_ours*.png')
        #import pdb; pdb.set_trace()
        cmd3 = ('ffmpeg -y -i '+pose_gif_path + ' -i ' + img_gif_path
                +' -filter_complex \"'+'[0:v][1:v]vstack' + '\"'
                + ' '+ comb_gif_path) 
        system(cmd1); system(cmd2); system(cmd3);

    print 'Results saved to '+save_path
    print 'Done.'

