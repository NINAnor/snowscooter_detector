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

def parseInputFiles(filesystem, input_path, workers, worker_idx, array_job=False):

    files = []
    include = ('.wav', '.flac', '.mp3', '.ogg', '.m4a', '.WAV', '.MP3')

    print("Worker {}".format(workers))
    print("Worker_idx {}".format(worker_idx))

    if array_job:
        for index, audiofile in enumerate(walk_audio(filesystem, input_path)):
            if index%workers == worker_idx:
                files.append(audiofile)
    else:
        for index, audiofile in enumerate(walk_audio(filesystem, input_path)):
            files.append(audiofile)

    files = [file for file in files if file.endswith(include)]
            
    print('Found {} files to analyze'.format(len(files)))

    return files

def initModel(model_path):
    model = CustomAudioCLIP(num_target_classes=2)
    model = model.load_from_checkpoint(model_path, num_target_classes=2)
    model.eval()
    #m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return model

def predict(testLoader, model, device):

    proba_list = []

    for array in testLoader:
        tensor = torch.tensor(array)
        tensor = tensor.to(device)
        output = model(tensor)
        output = np.exp(output.cpu().detach().numpy())
        proba_list.append(output[0])

    return proba_list

def get_outname(input, outfolder):

    # Get a name for the output // if there are multiple "." in the list
    # only remove the extension
    filename = input.split("/")[-1].split(".")
    if len(filename) > 2:
        filename = ".".join(filename[0:-1])
    else:
        filename = input.split("/")[-1].split(".")[0]

    # Make folder if it doesn't exist
    input_path = os.path.dirname(input)
    diff = os.path.relpath(input_path, outfolder).split("/")[2:]
    diff = "/".join(diff)

    outpath = os.sep.join([outfolder, diff])
    print(outpath)

    if len(outpath) > 0 and not os.path.exists(outpath):
        os.makedirs(outpath)

    outname = os.sep.join([outpath, filename + '.csv'])

    return outname

def write_results(prob_array, outname):

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

    with open(outname, 'w') as file:

        writer = csv.writer(file)
        header = ["start_detection", "end_detection", "label", "confidence"]

        writer.writerow(header)
        writer.writerows(rows_for_csv)

def analyzeFile(filesystem, file_path, model, out_folder, device, batch_size=1, num_workers=1):
    # Start time
    start_time = datetime.datetime.now()

    # Check if the output already exists
    outname = get_outname(file_path, out_folder)

    if os.path.exists(outname):
        print("File {} already exists".format(outname))
    else:
        # Run the predictions
        list_preds = AudioList().get_processed_list(filesystem, file_path)
        predLoader = DataLoader(list_preds, batch_size=batch_size, num_workers=num_workers, pin_memory=False)

        pred_array = predict(predLoader, model, device)
        write_results(pred_array, outname)

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

    parser.add_argument("--array_job",
                        help='Are you submitted an array job?',
                        default=False,
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Initiate model
    model = initModel(model_path=cfg["MODEL"], device=cfg["DEVICE"])

    # Do the connection to server
    print("Connecting to {}".format(cfg["CONNECTION_STRING"]))
    myfs = doConnection(cfg["CONNECTION_STRING"])

    file_list = parseInputFiles(myfs, cfg["INPUT_PATH"], cli_args.num_worker, cli_args.worker_index, array_job=cli_args.array_job)  

    flist = []
    for f in file_list:
        flist.append(f)
    print("Found {} files to analyze".format(len(flist)))

    # Analyze files
    for entry in flist:
        print("Analysing {}".format(entry))
        try:
            analyzeFile(myfs, entry, model, cfg["OUTPUT_PATH"],  device=cfg["DEVICE"], batch_size=1, num_workers=1)
        except:
            print("File {} failed to be analyzed".format(entry))