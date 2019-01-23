# MT-VAE for Multimodal Human Motion Synthesis

This is the code for ECCV 2018 paper [MT-VAE: Learning Motion Transformations to Generate Multimodal Human Dynamics](https://arxiv.org/abs/1808.04545) by Xinchen Yan, Akash Rastogi, Ruben Villegas, Kalyan Sunkavalli, Eli Shechtman, Sunil Hadap, Ersin Yumer, Honglak Lee.

<img src="https://sites.google.com/site/skywalkeryxc/multimodal_motion/00_comb_MTVAE.gif?attredirects=0" width="500px" height="200px"/>

Please follow the instructions to run the code.

## Requirements
MT-VAE requires or works with
* Mac OS X or Linux
* NVIDIA GPU

## Installing Dependency
* Install [TensorFlow](https://www.tensorflow.org/)
* Note: this implementation has been tested with [TensorFlow 1.3](https://www.tensorflow.org/versions/r1.3/).

## Data Preprocessing
* For Human3.6M dataset, please run the script to download the pre-processed dataset
```
bash prep_human36m_joints.sh
```
* Disclaimer: Please check the license of [Human3.6M dataset](http://vision.imar.ro/human3.6m/description.php) if you download this preprocessed version.

## Training (MT-VAE)
* If you want to train the MT-VAE human motion generator, please run the following script (usually it takes 1 day with a single Titan GPU).
```
bash demo_human36m_trainMTVAE.sh
```
* Alternatively, you can download the pre-trained MT-VAE model, please run the following script.
```
bash prep_human36m_model.sh
```

## Motion Synthesis Using Pre-trained MT-VAE Model
* Please run the following command to generate multiple diverse human motion given initial motion.
```
bash demo_human36m_inferMTVAE.sh
```

## Motion Analogy-making Using Pre-trained MT-VAE Model
* Please run the following command to execute motion analogy-making.
```
bash demo_human36m_analogyMTVAE.sh
```
## Hierchical Video Synthesis Using Pre-trained Image Generation Model
* Please download [full Human3.6M videos](http://vision.imar.ro/human3.6m/description.php) into the workspace/Human3.6M/ folder.
* We use a pre-trained model from the [ICML 2017 HierchVid Repository](https://github.com/rubenvillegas/icml2017hierchvid). Please run the following command for image synthesis given generated motion sequence.
```
CUDA_VISIBLE_DEVICE=0 python h36m_hierach_gensample.py
```
* Disclaimer: Please double check the license in that repository and cite [the paper](https://arxiv.org/abs/1704.05831) when use.


## Citation
If you find this useful, please cite our work as follows:
```
@inproceedings{yan2018mt,
  title={MT-VAE: Learning Motion Transformations to Generate Multimodal Human Dynamics},
  author={Yan, Xinchen and Rastogi, Akash and Villegas, Ruben and Sunkavalli, Kalyan and Shechtman, Eli and Hadap, Sunil and Yumer, Ersin and Lee, Honglak},
  booktitle={European Conference on Computer Vision},
  pages={276--293},
  year={2018},
  organization={Springer}
}
```

## Acknowledgements
We would like to thank the amazing developers and the open-sourcing community. Our implementation has especially been benefited from the following excellent repositories:
* Attribute2Image: [https://github.com/xcyan/eccv16_attr2img](https://github.com/xcyan/eccv16_attr2img)
* TensorFlow-PTN: [https://github.com/tensorflow/models/tree/master/research/ptn](https://github.com/tensorflow/models/tree/master/research/ptn)
* VideoGAN: [https://github.com/cvondrick/videogan](https://github.com/cvondrick/videogan)
* MoCoGAN: [https://github.com/sergeytulyakov/mocogan](https://github.com/sergeytulyakov/mocogan)
* HierchVid: [https://github.com/rubenvillegas/icml2017hierchvid](https://github.com/rubenvillegas/icml2017hierchvid)
* Sketch-RNN: [https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn)
* VRNN: [https://github.com/jych/nips2015_vrnn](https://github.com/jych/nips2015_vrnn)
* SVG: [https://github.com/edenton/svg](https://github.com/edenton/svg)


