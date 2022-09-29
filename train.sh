#!/usr/bin/bash 

rm -r mlruns
rm -r ray_experiment_0

DATA=$1 

docker run --rm -it --shm-size=10.02gb \
            -v $PWD:/app  \
            -v $DATA:/Data \
            --gpus all \
            audioclip:latest \
            poetry run python training/lightning_trainer/train_pipeline.py \
            --grid_search False