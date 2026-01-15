#!/bin/bash

scene_dir=$1
width=$2
height=$3
zNear=$4
zFar=$5
sensor_width=$6
(

    blender -b -P /home/zwr/code/depth_renderer/render_depth.py -- --set scene.workspace_path=${scene_dir} --set render.sensor_width=${sensor_width} --set render.width=${width} --set render.height=${height} --set render.zNear=${zNear} --set render.zFar=${zFar}
)