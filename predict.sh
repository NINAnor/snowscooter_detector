#!/usr/bin/bash 

DATA=$1 

docker run --rm -it --shm-size=10.02gb \
            -v $PWD:/app  \
            -v $DATA:/Data \
            --gpus all \
            audioclip:latest \
            poetry run python inference/predict.py 