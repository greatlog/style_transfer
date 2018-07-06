#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \ --output_dir ./experiment_gan/step1/ --summary_dir ./experiment_gan/step1/log --train_dir1 ./my_pic --train_dir2 ./gakii_pic --mode train --pre_trained_model False --vgg19_layer conv5_4 --input_size 512 --batch_size 4 --is_training True \


