# MT-VAE for human motion synthesis

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

## Data Preprocessing.
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

## Motion Synthesis Using Pre-trained Model.
* Please run the following command to generate multiple diverse human motion given initial motion.
```
bash demo_human36m_inferMTVAE.sh
```

## Motion Analogy-making Using Pre-trained Model.
* Please run the following command to execute motion analogy-making.
```
bash demo_human36m_analogyMTVAE.sh
```
