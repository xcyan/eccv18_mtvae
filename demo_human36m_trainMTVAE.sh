CUDA_VISIBLE_DEVICES=0 python h36m_train_mtvae.py \
  --inp_dir=workspace/ \
  --dataset_name=Human3.6M \
  --checkpoint_dir=checkpoints/ \
  --model_name=H36M_MTVAE \
  --max_number_of_steps=60000

