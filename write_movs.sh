#!/usr/bin/env bash

ARG1=${1:-'data/UA1Still-v0/new_rew/new_rew_s0'}

for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    ffmpeg -i "$ARG1/$i/$i.mp4" -c:v prores_ks -profile:v 3 "$ARG1/$i/$i.mov"
done
