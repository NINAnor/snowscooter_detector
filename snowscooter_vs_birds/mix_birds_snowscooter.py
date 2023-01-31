#!/usr/env python3

import argparse
from glob import glob
import os
import random
import soundfile

from audiomentations import Compose, AddBackgroundNoise
from utils.audio_processing import openAudioFile, splitSignal

def preprocess_with_one_sound(arr, sr, directory, min_db, max_db, p_sound=1):

    preprocess = Compose([
        AddBackgroundNoise(sounds_path=directory, 
        min_absolute_rms_in_db=min_db, 
        max_absolute_rms_in_db=max_db, 
        p=p_sound, 
        noise_rms = "absolute")
    ])
    processed_segment = preprocess(samples=arr, sample_rate=sr)
    return processed_segment

def preprocess_file(audio_path, noise_dir, min_db, max_db, length_segments=3):
 
    arr, sr = openAudioFile(audio_path)
    chunks = splitSignal(arr, sr, length_segments, 0, length_segments)

    # Randomly sample a chunk of "length_segments" seconds
    if len(chunks) > 1:
        r_number = random.randint(0, len(chunks)-1)
        r_chunk = chunks[r_number]
        mixed_array = preprocess_with_one_sound(r_chunk, sr, noise_dir, min_db, max_db)
        return mixed_array, r_chunk, sr
    else:
        print("The provided file is not long enough")

def save_processed_arrays(arr, sr, outname):
    soundfile.write(outname, arr, sr)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder_snowscooter",
                        help='Path to folder snowscooter',
                        default="/data/Model_training_data/Snowscooter_passages",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--folder_birds",
                        help='Path to folder birds',
                        default="/data/Noise_background/short_noises/lifeclef",
                        required=False,
                        type=str,
                        )
    parser.add_argument("--save_dir_mix",
                        help='Path to save folder',
                        default="/data/MIX_BIRDS_SNOWSCOOTER/audio_mix",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--save_dir_no_mix",
                        help='Path to save folder',
                        default="/data/MIX_BIRDS_SNOWSCOOTER/audio_snowscooter_only",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--l_mix",
                        help='length of the mixed clip',
                        default=3,
                        required=False,
                        type=int,
                        )

    parser.add_argument("--min_absolute_rms_in_db",
                        default=30.0,
                        required=False,
                        type=float,
                        )

    parser.add_argument("--max_absolute_rms_in_db",
                        default=8.0,
                        required=False,
                        type=float,
                        )

    parser.add_argument("--n_files",
            default=100,
            required=False,
            type=int,
            )   

    parser.add_argument("--n_iterations",
        default=1,
        required=False,
        type=int,
        )           
    cli_args = parser.parse_args()

    neg_min_db = - cli_args.min_absolute_rms_in_db
    neg_max_db = - cli_args.max_absolute_rms_in_db

    for it in range(cli_args.n_iterations):
        print(f"Iteration number: {it+1}")

        snowscooter_files = random.sample(glob(cli_args.folder_snowscooter + "/*.wav"), cli_args.n_files)

        for i, fpath in enumerate(snowscooter_files):
            # Create a name / output path for the file
            i = i * (it + 1)
            file_name = f"file_{i}.wav"
            outpath_mix = os.path.join(cli_args.save_dir_mix, file_name)
            outpath_no_mix = os.path.join(cli_args.save_dir_no_mix, file_name)

            # Create the file and save
            try:
                arr_mix, r_chunk, sr = preprocess_file(fpath, 
                    cli_args.folder_birds, 
                    neg_min_db,
                    neg_max_db,
                    cli_args.l_mix)

                save_processed_arrays(arr_mix, sr, outpath_mix)
                save_processed_arrays(r_chunk, sr, outpath_no_mix)
            except:
                print("Could not process the input file")

    # python snowscooter_vs_birds/mix_birds_snowscooter.py

                    