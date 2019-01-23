"""Hierarchical Video Generation Python Wrapper.

Code Adapted from Ruben Villegas's github project.
Please double check the license and cite the following paper when use:

Learning to Generate Long-term Future Hierarchical Prediction.
Ruben Villegas, Jimei Yang, Yuliang Zou, Sungryull Sohm, Xunyu Lin, and Honglak Lee. In ICML 2017.
"""
from image_synthesis import imagegen_model

import pickle

pred_path = 'checkpoints/gensample_comparison/MTVAE/'
pkl_file = open(pred_path + 'test.pkl', 'rb')
pred_data = pickle.load(pkl_file)
pkl_file.close()

net = imagegen_model.ImageAnalogyNet(
  pred_data['pred_data'][0] - pred_data['ob_start'][0],
  pred_data['pred_keypoints'][0].shape[1])

pose_path = 'checkpoints/gensample_comparison/MTVAE/'
save_path = 'checkpoints/gensample_comparison/MTVAE_imgs/'

net.run_inference(pose_path, 'MTVAE', pred_data, save_path)

