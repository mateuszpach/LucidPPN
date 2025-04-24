#!/bin/bash
#SBATCH --job-name=unambiguous-prototypes
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxh100
#SBATCH --qos=quick

eval "$(conda shell.bash hook)"
conda activate unambiguous-prototypes
cd /home/z1164034/unambiguous-prototypes/PIPNet

python main.py \
  --dataset CUB-200-2011 \
  --validation_size 0.0 \
  --net convnext_tiny_26 \
  --batch_size 64 \
  --batch_size_pretrain 128 \
  --epochs 60 \
  --optimizer Adam \
  --lr 0.05 \
  --lr_block 0.0005 \
  --lr_net 0.0005 \
  --weight_decay 0.0 \
  --log_dir ./runs/pipnet_cub \
  --num_features 0 \
  --image_size 224 \
  --freeze_epochs 10 \
  --dir_for_saving_images Visualization_results \
  --epochs_pretrain 10 \
  --seed 1 \
  --num_workers 8 \
  --option 5

#python main.py \
#--dataset DOGS \
#--validation_size 0.0 \
#--net convnext_tiny_26 \
#--batch_size 64 \
#--batch_size_pretrain 128 \
#--epochs 60 \
#--optimizer Adam \
#--lr 0.05 \
#--lr_block 0.0005 \
#--lr_net 0.0005 \
#--weight_decay 0.0 \
#--log_dir ./runs/pipnet_dogs \
#--num_features 0 \
#--image_size 224 \
#--freeze_epochs 10 \
#--dir_for_saving_images Visualization_results \
#--epochs_pretrain 10 \
#--seed 1 \
#--num_workers 8 \

#python main.py \
#--dataset FLOWERS \
#--validation_size 0.0 \
#--net convnext_tiny_26 \
#--batch_size 64 \
#--batch_size_pretrain 128 \
#--epochs 60 \
#--optimizer Adam \
#--lr 0.05 \
#--lr_block 0.0005 \
#--lr_net 0.0005 \
#--weight_decay 0.0 \
#--log_dir ./runs/pipnet_flowers \
#--num_features 0 \
#--image_size 224 \
#--freeze_epochs 10 \
#--dir_for_saving_images Visualization_results \
#--epochs_pretrain 10 \
#--seed 1 \
#--num_workers 8 \


