#!/usr/bin/env bash

ARG1=${1:-'data/UA1Still-v0/new_rew/new_rew_s0'}

for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    # framerate 50 based on 0.02 DMC control_timestep
    ffmpeg -framerate 50 -pattern_type glob -i "$ARG1/$i/*.jpg" -c:v libx264 -pix_fmt yuv420p "$ARG1/$i/$i.mp4"
done
