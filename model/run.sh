#!/bin/bash
CUDA_VISIBLE_DEVICES="1" python Run.py --dataset COVID_NY --test_ratio 0.2 --model_type SRE_v5 --use_mask False --use_smooth_stats True --seed 1 --epochs 100 --lr_init 1e-3 --batch_size 16 --lag 7 --horizon 7 --lstm_num_layers 1 --early_stop True --embed_dim 32 --state "New York" --num_nodes 1500 --input_dim 768
