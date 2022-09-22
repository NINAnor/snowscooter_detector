import argparse
import datetime
import numpy as np
import torch
import os
import glob
import yaml
import csv
import fs

from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
from yaml.loader import FullLoader
from multiprocessing import Pool
from fs.sshfs import SSHFS

from utils.utils_inference import AudioList

def doConnection(host, user, password):

    myfs = SSHFS(
        host=host, user=user, passwd=password, pkey=None, timeout=20, port=22,
        keepalive=10, compress=False, config_path='~/.ssh/config')
    print("Connection to the input folder has been successfully made")
    return myfs

def walk_audio(filesystem, input_path):
    # Get all files in directory with os.walk
    walker = filesystem.walk(input_path, filter=['*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a', '*.WAV', '*.MP3'])
    for path, dirs, flist in walker:
        for f in flist:
            yield fs.path.combine(path, f.name)

def parseInputFiles(filesystem, input_path, workers, worker_idx, array_job=False):

    print("Worker {}".format(workers))
    print("Worker_idx {}".format(worker_idx))

    files = []

    if array_job:
        for index, audiofile in enumerate(walk_audio(filesystem, input_path)):
            if index%workers == worker_idx:
                files.append(audiofile)
    else:
        for index, audiofile in enumerate(walk_audio(filesystem, input_path)):
            files.append(audiofile)
            
    print('Found {} files to analyze'.format(len(files)))

    return files

def initModel(model_path):
    m = torch.load(model_path).eval()
    m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return m_q

def predict(testLoader, model):

    proba_list = []

    for array in testLoader:
        tensor = torch.tensor(array)
        output = model(tensor)
        output = np.exp(output.detach().numpy())
        proba_list.append(output[0][1])

    return proba_list

def write_results(prob_array, input, output):

    # Store the array result in a CSV friendly format
    rows_for_csv = []
    idx_begin = 0

    for item in prob_array:
        # Get the properties of the detection (start, end, label and confidence)
        idx_end = idx_begin + 3
        arr = np.array(item)
        label = np.argmax(arr, axis=0)
        max_value = arr.max()

        # Map the label index to its category

        # If the label is not "soundscape" then write the row:
        if label != 0:
            item_properties = [idx_begin, idx_end, label, max_value]
            rows_for_csv.append(item_properties)

        # Update the start time of the detection
        idx_begin = idx_end

    # Get a name for the output // if there are multiple "." in the list
    # only remove the extension
    filename = input.split("/")[-1].split(".")
    if len(filename) > 2:
        filename = ".".join(filename[0:-1])
    else:
        filename = input.split("/")[-1].split(".")[0]

    outname = os.sep.join([output, filename + '.csv'])

    with open(outname, 'w') as file:

        writer = csv.writer(file)
        header = ["start_detection", "end_detection", "label", "confidence"]

        writer.writerow(header)
        writer.writerows(rows_for_csv)

def analyzeFile(filesystem, file_path, model, out_folder, batch_size=1, num_workers=1):
    # Start time
    start_time = datetime.datetime.now()

    # Run the predictions
    list_preds = AudioList().get_processed_list(filesystem, file_path)
    predLoader = DataLoader(list_preds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

    pred_array = predict(predLoader, model)
    write_results(pred_array, file_path, out_folder)

    # Give the tim it took to analyze file
    delta_time = (datetime.datetime.now() - start_time).total_seconds()
    print('Finished {} in {:.2f} seconds'.format(file_path, delta_time), flush=True)

if __name__ == "__main__":

    # Get the config for doing the predictions
   # FOR TESTING THE PIPELINE WITH ONE FILE
    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help='Path to the config file',
                        default="/app/config_inference.yaml",
                        required=False,
                        type=str,
                        )

    parser.add_argument("--num_worker",
                        help='Path to the config file',
                        default=1,
                        required=False,
                        type=int,
                        )

    parser.add_argument("--worker_index",
                        help='Path to the config file',
                        default=1,
                        required=False,
                        type=int,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Initiate model
    model = initModel(model_path=cfg["MODEL"])

    # Do the connection to server
    myfs = doConnection(host=cfg["HOST"], user=cfg["USER"], password=cfg["PASSWORD"])
    if not myfs:  # Nothing available
        exit(0)

    file_list = parseInputFiles(myfs, cfg["INPUT_PATH"], cli_args.num_worker, cli_args.worker_index)  

    flist = []
    for f in file_list:
        flist.append(f)
    print("Found {} files to analyze".format(len(flist)))

    # Analyze files
    if cfg["CPU_THREADS"] < 2:
        for entry in flist:
            #try:
            analyzeFile(myfs, entry, model, cfg["OUTPUT_PATH"], batch_size=1, num_workers=1)
            #except:
            #    print("File {} failed to be analyzed".format(entry))
    else:
        with Pool(cfg["CPU_THREADS"]) as p:
            p.map(analyzeFile, myfs, flist, model, cfg["OUTPUT_PATH"], batch_size=1, num_workers=1)

# docker run --rm -it -v $pwd:/app/ -v ~/Data/:/Data registry.gitlab.com/nina-data/audioclip:latest poetry run python prediction_scripts/predict.py