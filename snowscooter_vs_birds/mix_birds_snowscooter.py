#!/usr/env python3

import argparse
import os
import random

from pydub import AudioSegment
from glob import glob

def mix_sounds(folder1, folder2, save_dir, l_mix=3000):

    snowscooter_files = random.sample(glob(folder1 + "/*.wav"), 100)
    bird_files = random.sample(glob(folder2 + "/*.wav"), 100)

    i=0

    for path1, path2 in zip(snowscooter_files, bird_files):

        i=i+1
        sound1 = AudioSegment.from_file(path1, format="wav").set_channels(1)
        sound2 = AudioSegment.from_file(path2, format="wav").set_channels(1)

        # Overlay sound2 over sound1 at position 0  (use louder instead of sound1 to use the louder version)
        overlay = sound1.overlay(sound2, position=0)
        overlay = overlay[0:l_mix]

        # simple export
        file_name = f"file_{i}.wav"
        outpath = os.path.join(save_dir, file_name)
        overlay.export(outpath, format="wav")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--folder1",
                        help='Path to folder 1',
                        default="/data//Model_training_data/Snowscooter_passages",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--folder2",
                        help='Path to folder 2',
                        default="/data/Noise_background/short_noises/lifeclef",
                        required=False,
                        type=str,
                        )
    parser.add_argument("--save_dir",
                        help='Path to save folder',
                        default="/data/MIX_BIRDS_SNOWSCOOTER",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--l_mix",
                        help='length of the mixed clip',
                        default=3000,
                        required=False,
                        type=int,
                        )
    cli_args = parser.parse_args()

    mix_sounds(cli_args.folder1, cli_args.folder2, cli_args.save_dir, cli_args.l_mix)

    # ./snowscooter_vs_birds/mix_birds_snowscooter.py

                    