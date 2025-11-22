#!/bin/bash

env_name=AR_Backend_Outdoor
script_path=/home/zwr/code/vggt/demo_colmap.py
scene_dir=$1

CUDA_VISIBLE_DEVICES=2 conda run -n  ${env_name} python ${script_path} --scene_dir ${scene_dir}
