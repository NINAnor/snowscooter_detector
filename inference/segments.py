import glob
import os
import argparse
import traceback
import numpy as np
import yaml
import fs
import tqdm
import sqlalchemy
import pandas as pd

from yaml import FullLoader

from utils.audio_processing import openCachedFile, openAudioFile, saveSignal
from utils.parsing_utils import remove_extension

def read_db(db_path, query, n_segments):

    # Initiate and connect to the engine
    engine=sqlalchemy.create_engine(db_path)
    connection=engine.connect()

    # Filter the results given the query and store in a pandas DF
    df = pd.read_sql_query(query, connection)
    print("A total of {} detections have been found".format(len(df)))

    # Subsample pandas DF for extracting n_segments
    sampled_df = df.sample(n_segments)

    return sampled_df

def doConnection(connection_string):

    if connection_string is False:
        myfs = False
    else:
        myfs = fs.open_fs(connection_string)
    return myfs

def walk_audio(filesystem, input_path):
    # Get all files in directory with os.walk
    if filesystem:
        walker = filesystem.walk(input_path, filter=['*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a', '*.WAV', '*.MP3'])
        for path, dirs, flist in walker:
            for f in flist:
                yield fs.path.combine(path, f.name)
    else:
        for path, dirs, flist in os.walk(input_path):
            for f in flist:
                yield os.path.join(path, f)

def parseAudioFiles(filesystem, apath):

    afiles = []

    for index, afile in enumerate(walk_audio(filesystem, apath)):
        afiles.append(afile)

    return afiles

def openAudio(afile, sample_rate, filesystem):

    if not filesystem:
        sig, rate = openAudioFile(afile, sample_rate)
    else:
        sig, rate = openCachedFile(filesystem, afile, sample_rate)

    return sig, rate

def extract_segment(afile, df, filesystem, out_path, sample_rate=44100):

    df['filename'] = df['filename'].apply(remove_extension)
    results = df[df['filename'].isin([remove_extension(afile)])]

    # There's a match between some rows and the audiofile
    if len(results) > 0:
        print('Found {} segments in {}'.format(len(results), afile))

        # Could be multiple for for an audiofile
        for index, row in results.iterrows():
            filename = row['filename']
            start = row['start_detection']
            end = row['end_detection']
            conf = row['conf_agg']
            hr = row['hr_agg']

            sig, rate = openAudio(afile, sample_rate, filesystem)

            # Get segment raw audio from signal
            seg_sig = sig[int(start):int(end)]

            # Make output path
            outpath = os.sep.join([out_path, os.path.dirname(afile)])
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)

            # Save segment
            seg_name = 'start={}_end={}_conf={:.3f}_hr={:.3f}_{}.wav'.format(start, end, conf, hr, filename.split(os.sep)[-1].rsplit('.', 1)[0])
            seg_path = os.path.join(outpath, seg_name)
            saveSignal(seg_sig, seg_path)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="/app/config_inference.yaml",
                        required=False,
                        type=str,
                        )
    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Do the connection to server
    myfs = doConnection(cfg["CONNECTION_STRING"])

    # Read the database and convert into a pandas dataframe
    df = read_db(cfg['DB_PATH'], cfg['QUERY'], cfg['NUM_SEGMENTS'])

    # Parse the files
    parsedfiles = parseAudioFiles(myfs, cfg["INPUT_PATH"])

    # Extract segments   
    for entry in tqdm.tqdm(parsedfiles):
        extract_segment(entry, df, myfs, cfg["OUT_PATH_SEGMENTS"])