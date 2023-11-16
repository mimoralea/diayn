for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    # framerate 50 based on 0.02 DMC control_timestep
    ffmpeg -framerate 50 -pattern_type glob -i "data/UA1Still-v0/new_rew/new_rew_s0/$i/*.jpg" -c:v libx264 -pix_fmt yuv420p "data/UA1Still-v0/new_rew/new_rew_s0/$i/$i.mp4"
done
