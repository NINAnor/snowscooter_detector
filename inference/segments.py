import glob
import os
import argparse
import traceback
import numpy as np
import yaml

from yaml import FullLoader

from utils import audio_processing

def remove_extension(input):

    filename = input.split("/")[-1].split(".")

    if len(filename) > 2:
        filename = ".".join(filename[0:-1])
    else:
        filename = input.split("/")[-1].split(".")[0]

    return filename

def parseFolders(apath, rpath):

    audio_files = [f for f in glob.glob(apath + "/**/*", recursive = True) if os.path.isfile(f)]
    #audio_fixed = [f for f in audio_files if f.endswith(".WAV", ".wav", ".mp3")]
    audio_no_extension = []
    for audio_file in audio_files:
        audio_file_no_extension = remove_extension(audio_file)
        audio_no_extension.append(audio_file_no_extension)


    result_files = [f for f in glob.glob(rpath + "/**/*", recursive = True) if os.path.isfile(f)]
    #result_fixed = [f for f in result_files if f.endswith(".csv")]

    flist = []
    for result in result_files:
        result_no_extension = remove_extension(result)
        print(result_no_extension)
        is_in = result_no_extension in audio_no_extension

        if is_in:
            audio_idx = audio_no_extension.index(result_no_extension)
            pair = {'audio': audio_files[audio_idx], 'result': result}
            flist.append(pair)
        else:
            continue

    print('Found {} audio files with valid result file.'.format(len(flist)))

    return flist

def parseFiles(flist, max_segments=10, threshold=0.6):

    species_segments = {}
    for f in flist:

        # Paths
        afile = f['audio'] 
        rfile = f['result']

        # Get all segments for result file
        segments = findSegments(afile, rfile, threshold)

        # Parse segments by species
        for s in segments:
            if s['label'] not in species_segments:
                species_segments[s['label']] = []
            species_segments[s['label']].append(s)

    # Shuffle segments for each species and limit to max_segments
    for s in species_segments:
        np.random.shuffle(species_segments[s])
        species_segments[s] = species_segments[s][:max_segments]

    # Make dict of segments per audio file
    segments = {}
    seg_cnt = 0
    for s in species_segments:
        for seg in species_segments[s]:
            if not seg['audio'] in segments:
                segments[seg['audio']] = []
            segments[seg['audio']].append(seg)
            seg_cnt += 1

    print('Found {} segments in {} audio files.'.format(seg_cnt, len(segments)))

    # Convert to list
    flist = []
    for f in segments:
        flist.append((f, segments[f]))

    return flist

def findSegments(afile, rfile, confidence_thr):

    segments = []

    # Open and parse result file
    lines = []
    with open(rfile, 'r') as rf:
        for line in rf.readlines():
            lines.append(line.strip())

    # Auto-detect result type
    rtype = rfile.split("/")[-1].split(".")[-1]

    for i in range(len(lines)):
        if rtype == 'csv' and i > 0:

            d = lines[i].split(',')
            start = float(d[0])
            end = float(d[1])
            label = d[2]
            confidence = float(d[3])

            # Check if confidence is high enough
            if confidence >= confidence_thr:
                segments.append({'audio': afile, 'start': start, 'end': end, 'label': label, 'confidence': confidence})

    return segments

def extractSegments(item, sample_rate, out_path, seg_length=3):

    # Paths and config
    afile = item[0]
    segments = item[1]
    seg_length = seg_length

    # Status
    print('Extracting segments from {}'.format(afile))

    # Open audio file
    sig, rate = audio_processing.openAudioFile(afile, sample_rate)

    # Extract segments
    seg_cnt = 1
    for seg in segments:

        try:

            # Get start and end times
            start = int(seg['start'] * sample_rate)
            end = int(seg['end'] * sample_rate)
            offset = ((seg_length * sample_rate) - (end - start)) // 2
            start = max(0, start - offset)
            end = min(len(sig), end + offset)  

            # Make sure segmengt is long enough
            if end > start:

                # Get segment raw audio from signal
                seg_sig = sig[int(start):int(end)]

                # Make output path
                outpath = os.path.join(out_path, seg['label'])
                if not os.path.exists(outpath):
                    os.makedirs(outpath, exist_ok=True)

                # Save segment
                seg_name = 'start={}_end={}_conf={:.3f}_{}.wav'.format(seg['start'], seg['end'], seg['confidence'], seg['audio'].split(os.sep)[-1].rsplit('.', 1)[0])
                seg_path = os.path.join(outpath, seg_name)
                audio_processing.saveSignal(seg_sig, seg_path)
                seg_cnt += 1

        except:

            # Print traceback
            print(traceback.format_exc(), flush=True)

            # Write error log
            msg = 'Error: Cannot extract segments from {}.\n{}'.format(afile, traceback.format_exc())
            print(msg, flush=True)
            #writeErrorLog(msg)
            break

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="/app/prediction_scripts/config.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Parse audio and result folders
    parsed_folders = parseFolders(cfg["INPUT_PATH"], cfg["OUTPUT_PATH"])

    # Set confidence threshold
    MIN_CONFIDENCE = float(cfg["MIN_CONFIDENCE"])

    # Parse file list and make list of segments
    parsed_files = parseFiles(parsed_folders, cfg["NUM_SEGMENTS"], cfg["THRESHOLD"])

    # Add config items to each file list entry.
    flist = []
    for entry in parsed_files:
        flist.append(entry)
    
    # Extract segments   
    for entry in flist:
        extractSegments(entry, cfg["SAMPLE_RATE"], cfg["OUT_PATH_SEGMENTS"])

# docker run --rm -it -v $pwd:/app/ -v ~/Data/:/Data registry.gitlab.com/nina-data/audioclip:latest poetry run python prediction_scripts/segments.py
