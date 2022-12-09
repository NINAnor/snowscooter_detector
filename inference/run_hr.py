import argparse
import os
import numpy as np
import yaml
from yaml import FullLoader
import librosa

from tqdm import tqdm

from utils.audio_signal import AudioSignal
from utils.audio_processing import saveSignal

def walk_audio(input_path):
    for path, dirs, flist in os.walk(input_path):
        for f in flist:
            yield os.path.join(path, f)

def parseFolders(spath, workers, worker_idx, array_job=False):

    files = []
    include = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.WAV', '.MP3')

    print("Worker {}".format(workers))
    print("Worker_idx {}".format(worker_idx))

    if array_job:
        for index, audiofile in enumerate(walk_audio(spath)):
            if index%workers == worker_idx:
                files.append(audiofile)
    else:
        for index, audiofile in enumerate(walk_audio(spath)):
            files.append(audiofile)

    files = [file for file in files if file.endswith(include)]        
    print('Found {} files to analyze'.format(len(files)))

    return files

def compute_hr(samples):

    signal = AudioSignal(samples = samples, fs=44100)

    signal.apply_butterworth_filter(
        order=18, Wn=np.asarray([1, 600]) / (signal.fs / 2)
    )
    signal_hr = signal.harmonic_ratio(
        win_length=int(1 * signal.fs),
        hop_length=int(0.1 * signal.fs),
        window="hamming",
    )
    hr = np.mean(signal_hr)
    
    return hr

def save_segment(segment_filename, samples, hr, save_path, threshold):

    if hr > threshold:
        # Make output path
        outpath = os.sep.join([save_path, os.path.dirname(segment_filename)])
        if not os.path.exists(outpath):
            os.makedirs(outpath, exist_ok=True)

        # Save segment
        seg_name = segment_filename.split("/")[-1].split(".wav")[0] + '_hr={:.3}.wav'.format(hr)
        seg_path = os.path.join(outpath, seg_name)
        saveSignal(samples, seg_path)
    else:
        print("The harmonic ratio for {} is less than {} which means that the file probably contains wind".format(segment_filename, threshold))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="config_inference.yaml",
                        required=False,
                        type=str,
                        )
    parser.add_argument('--workers', type=int, default=1, help='Number of workers')
    parser.add_argument('--worker_index', type=int, default=1, help='Worker index')
    parser.add_argument("--array_job", help='Are you submitted an array job?', default=False, required=False, type=str)
    
    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Find all the segments to be processed by the HR algorithm
    segments = parseFolders(cfg["OUT_PATH_SEGMENTS"], cli_args.workers, cli_args.worker_index)

    # Loop through all the files
    for entry in tqdm(segments):
        sig, rate = librosa.load(entry, sr=44100, mono=True, res_type='kaiser_fast')
        hr = compute_hr(sig)
        save_segment(entry, sig, hr, cfg["OUT_PATH_SEGMENTS_AFTER_HR"], cfg["THRESHOLD_HR"])