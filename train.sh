#!/usr/bin/bash 

rm -r mlruns
rm -r ray_experiment_0

#sed -i "s/experiment_2/experiment_3/g" config.yaml

docker run --rm -it --shm-size=1.02gb -v ~/Code/AudioCLIP:/app -v ~/Data/:/Data \
            --gpus all audioclip:latest \
            poetry run python lightning_trainer/train_pipeline.py \
            --grid_search False

