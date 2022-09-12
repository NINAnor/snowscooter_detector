#!/usr/bin/bash 
docker run --rm -it --gpus all -v ~/Code/AudioCLIP:/app/ audioclip:latest poetry run python prediction_scripts/predict.py 

#docker run --rm -it -v ~/Code/AudioCLIP:/app -v ~/Data/:/Data --gpus all audioclip:latest poetry run python prediction_scripts/predict_one_file.py \
#                    --input /Data/soundsolution/anthropogenic/birds/Bergfink-1-par-varnar-for-lavskrika-Rymdcampus-Kiruna-Lp-2010-06-27.mp3 \
#                    --output /Data/soundsolution/pipeline_out/pred_json/
