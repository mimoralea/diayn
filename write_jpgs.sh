#!/usr/bin/env bash

ARG1=${1:-'data/UA1Still-v0/new_rew/new_rew_s0'}
ARG2=${2:-'UA1Still-v0'}

for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    python diayn/spinningup/spinup/utils/test_policy.py "$ARG1" --env_id "$ARG2" --norender --write --skill "$i"
done
