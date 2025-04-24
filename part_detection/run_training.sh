#!/bin/bash
#SBATCH --job-name=unambiguous-prototypes
#SBATCH --nodes=1
#SBATCH --mem=40G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=dgxa100
#SBATCH --qos=quick

#module load GCCcore/11.2.0
#module load Python/3.9.6
#module load CUDA/12.1.1
#module load Miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate unambiguous-prototypes
cd /home/z1164034/unambiguous-prototypes/part_detection

datasets=("flowers" "dogs" "cars" "cub")
num_parts=(2 3 4 8 12)

for dataset in "${datasets[@]}"; do
    for parts in "${num_parts[@]}"; do
        python main.py --model_name "${dataset}_${parts}parts" \
            --data_root /home/z1164034/datasets/ \
            --dataset "$dataset" --num_parts "$parts" --batch_size 16 --image_size 448 --epochs 28 --save_figures

        python save_maps.py --model_name "${dataset}_${parts}parts" \
            --data_root /home/z1164034/datasets/ \
            --dataset "$dataset" --num_parts "$parts" --image_size 448 \
            --pretrained_model_path "./$dataset/${dataset}_${parts}parts.pt" \
            --save_root "/home/z1164034/datasets/${dataset}_${parts}parts/"
    done
done

