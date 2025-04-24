#!/bin/bash
#SBATCH --job-name=unambiguous-prototypes
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --qos=quick

eval "$(conda shell.bash hook)"
conda activate unambiguous-prototypes
cd /home/z1164034/unambiguous-prototypes/MetiNet

## Analysis: part_weights
#datasets=("CUB")
#num_parts=(12)
#part_weights=(0.0 0.2 0.4 0.6 0.8 1.0 1.2 1.4 1.6 1.8 2.0)
#
#declare -A num_classes=( ["CUB"]=200 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#        for weight in "${part_weights[@]}"; do
#            python main.py \
#                --dataset "$dataset" \
#                --num_classes "$classes" \
#                --net convnext_tiny_26 \
#                --batch_size 64 \
#                --epochs 40 \
#                --freeze_epochs 15 \
#                --aggregate mean \
#                --no_color_epochs 0 \
#                --optimizer Adam \
#                --lr_class 0.5 \
#                --lr_net 0.002 \
#                --lr_color 0.002 \
#                --weight_decay 0.0 \
#                --part_weight "$weight" \
#                --proto_class_weight 1.0 \
#                --color_class_weight 1.0 \
#                --log_dir "./runs/null" \
#                --num_parts "$parts" \
#                --image_size 224 \
#                --dir_for_saving_images Visualization_results \
#                --seed 1 \
#                --num_workers 8
#        done
#    done
#done

## Analysis: no_color_epochs
#datasets=("CUB")
#num_parts=(12)
#no_color_epochs=(0 10 20 30 40)
#
#declare -A num_classes=( ["CUB"]=200 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#        for no_color_epoch in "${no_color_epochs[@]}"; do
#            python main.py \
#                --dataset "$dataset" \
#                --num_classes "$classes" \
#                --net convnext_tiny_26 \
#                --batch_size 64 \
#                --epochs 40 \
#                --freeze_epochs 15 \
#                --aggregate mean \
#                --no_color_epochs "$no_color_epoch" \
#                --optimizer Adam \
#                --lr_class 0.5 \
#                --lr_net 0.002 \
#                --lr_color 0.002 \
#                --weight_decay 0.0 \
#                --part_weight 1.4 \
#                --proto_class_weight 1.0 \
#                --color_class_weight 1.0 \
#                --log_dir "./runs/metinet_${dataset}${parts}_${no_color_epoch}" \
#                --num_parts "$parts" \
#                --image_size 224 \
#                --dir_for_saving_images Visualization_results \
#                --seed 1 \
#                --num_workers 8
#        done
#    done
#done

## Analysis: strong_hue_augmentation
#datasets=("CUB")
#num_parts=(12)
#
#declare -A num_classes=( ["CUB"]=200 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#          python main.py \
#              --dataset "$dataset" \
#              --num_classes "$classes" \
#              --net convnext_tiny_26 \
#              --batch_size 64 \
#              --epochs 40 \
#              --freeze_epochs 15 \
#              --aggregate mean \
#              --no_color_epochs 0 \
#              --optimizer Adam \
#              --lr_class 0.5 \
#              --lr_net 0.002 \
#              --lr_color 0.002 \
#              --weight_decay 0.0 \
#              --part_weight 1.4 \
#              --proto_class_weight 1.0 \
#              --color_class_weight 1.0 \
#              --log_dir "./runs/null" \
#              --num_parts "$parts" \
#              --image_size 224 \
#              --dir_for_saving_images Visualization_results \
#              --seed 1 \
#              --num_workers 8 \
#              --strong_hue_augmentation 0.5 \
#              --state_dict_dir_net "./runs/metinet_${dataset}${parts}/checkpoint_epoch40"
#    done
#done

## Analysis: crop_augmentation
#datasets=("CUB")
#num_parts=(12)
#crops=(0.25 0.5 0.75 1.0)
#
#declare -A num_classes=( ["CUB"]=200 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#        for crop in "${crops[@]}"; do
#            python main.py \
#                --dataset "$dataset" \
#                --num_classes "$classes" \
#                --net convnext_tiny_26 \
#                --batch_size 64 \
#                --epochs 40 \
#                --freeze_epochs 15 \
#                --aggregate mean \
#                --no_color_epochs 0 \
#                --optimizer Adam \
#                --lr_class 0.5 \
#                --lr_net 0.002 \
#                --lr_color 0.002 \
#                --weight_decay 0.0 \
#                --part_weight 1.4 \
#                --proto_class_weight 1.0 \
#                --color_class_weight 1.0 \
#                --log_dir "./runs/null" \
#                --num_parts "$parts" \
#                --image_size 224 \
#                --dir_for_saving_images Visualization_results \
#                --seed 1 \
#                --num_workers 8 \
#                --crop_augmentation "$crop" \
#                --state_dict_dir_net "./runs/metinet_${dataset}${parts}/checkpoint_epoch40"
#        done
#    done
#done


# Main results
datasets=("CUB" "CARS" "DOGS" "FLOWERS")
num_parts=(2 3 4 8 12)

declare -A num_classes=( ["CARS"]=196 ["DOGS"]=120 ["CUB"]=200 ["FLOWERS"]=102 )

for dataset in "${datasets[@]}"; do
    classes="${num_classes[$dataset]}"
    for parts in "${num_parts[@]}"; do
        python main.py \
            --dataset "$dataset" \
            --num_classes "$classes" \
            --net convnext_tiny_26 \
            --batch_size 64 \
            --epochs 40 \
            --freeze_epochs 15 \
            --aggregate mean \
            --no_color_epochs 0 \
            --optimizer Adam \
            --lr_class 0.5 \
            --lr_net 0.002 \
            --lr_color 0.002 \
            --weight_decay 0.0 \
            --part_weight 1.4 \
            --proto_class_weight 1.0 \
            --color_class_weight 1.0 \
            --log_dir "/shared/results/z1164034/MetiNet/runs/metinet_${dataset}${parts}" \
            --num_parts "$parts" \
            --image_size 224 \
            --dir_for_saving_images Visualization_results \
            --seed 3 \
            --num_workers 8
    done
done

## Other visualization
##datasets=("CUB")
##num_parts=(4)
##datasets=("FLOWERS")
##num_parts=(2)
#datasets=("CARS" "DOGS")
#num_parts=(4)
#
#declare -A num_classes=( ["CARS"]=196 ["DOGS"]=120 ["CUB"]=200 ["FLOWERS"]=102 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#        python main.py \
#            --dataset "$dataset" \
#            --num_classes "$classes" \
#            --net convnext_tiny_26 \
#            --batch_size 64 \
#            --epochs 40 \
#            --freeze_epochs 15 \
#            --aggregate mean \
#            --no_color_epochs 0 \
#            --optimizer Adam \
#            --lr_class 0.5 \
#            --lr_net 0.002 \
#            --lr_color 0.002 \
#            --weight_decay 0.0 \
#            --part_weight 1.4 \
#            --proto_class_weight 1.0 \
#            --color_class_weight 1.0 \
#            --log_dir "./runs/${dataset}${parts}_vis" \
#            --num_parts "$parts" \
#            --image_size 224 \
#            --dir_for_saving_images Visualization_results \
#            --seed 1 \
#            --num_workers 8 \
#            --state_dict_dir_net "./runs/metinet_${dataset}${parts}/checkpoint_epoch40"
#    done
#done

## Second user study
#datasets=("CUB")
#num_parts=(4)
#
#declare -A num_classes=( ["CARS"]=196 ["DOGS"]=120 ["CUB"]=200 ["FLOWERS"]=102 )
#
#for dataset in "${datasets[@]}"; do
#    classes="${num_classes[$dataset]}"
#    for parts in "${num_parts[@]}"; do
#        python main.py \
#            --dataset "$dataset" \
#            --num_classes "$classes" \
#            --net convnext_tiny_26 \
#            --batch_size 64 \
#            --epochs 40 \
#            --freeze_epochs 15 \
#            --aggregate mean \
#            --no_color_epochs 0 \
#            --optimizer Adam \
#            --lr_class 0.5 \
#            --lr_net 0.002 \
#            --lr_color 0.002 \
#            --weight_decay 0.0 \
#            --part_weight 1.4 \
#            --proto_class_weight 1.0 \
#            --color_class_weight 1.0 \
#            --log_dir "/shared/results/z1164034/MetiNet/runs/${dataset}${parts}_vis_2nd_neq" \
#            --num_parts "$parts" \
#            --image_size 224 \
#            --dir_for_saving_images Visualization_results \
#            --seed 1 \
#            --num_workers 8 \
#            --state_dict_dir_net "/shared/results/z1164034/MetiNet/runs/metinet_${dataset}${parts}/checkpoint_epoch40"
#    done
#done