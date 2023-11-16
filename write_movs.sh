for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    ffmpeg -i "data/UA1Still-v0/new_rew/new_rew_s0/vids/$i.mp4" -c:v prores_ks -profile:v 3 "data/UA1Still-v0/new_rew/new_rew_s0/vids/$i.mov"
done
