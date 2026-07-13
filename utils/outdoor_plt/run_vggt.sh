#!/bin/bash

env_name=AR_Backend_Outdoor
script_path=/home/zwr/code/vggt/demo_colmap.py
scene_dir=$1

(
    export CUDA_HOME=/usr/local/cuda-11.8
    export PATH=/usr/local/cuda-11.8/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-11.8/lib64:$LD_LIBRARY_PATH

    CUDA_VISIBLE_DEVICES=2 conda run -n  ${env_name} python ${script_path} --scene_dir ${scene_dir}
)