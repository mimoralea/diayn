for ((i=0; i<=19; i++))
do
    echo "Exporting skill $i"
    python diayn/spinningup/spinup/utils/test_policy.py "data/UA1Still-v0/new_rew/new_rew_s0" --env_id "UA1Still-v0" --norender --write --skill "$i"
done
