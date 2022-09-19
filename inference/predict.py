import argparse
import datetime
import numpy as np
import torch
import os
import glob
import yaml
import csv

from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
from yaml.loader import FullLoader

from utils.utils_inference import AudioList

def initModel(model_path):
    m = torch.load(model_path).eval()
    m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return m_q
    
def getPredLoader(input):
    list_preds = AudioList().get_processed_list(input)
    predLoader = DataLoader(list_preds, batch_size=1, num_workers=4, pin_memory=False)
    return predLoader

def predict(testLoader, model):

    proba_list = []

    for array in testLoader:
        tensor = torch.tensor(array)
        output = model(tensor)
        output = np.exp(output.detach().numpy())
        proba_list.append(output[0])

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

def analyzeFile(file_path, model, out_folder):
    # Start time
    start_time = datetime.datetime.now()

    # Run the predictions
    predLoader = getPredLoader(file_path)
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
                        default="/app/prediction_scripts/config.yaml",
                        required=False,
                        type=str,
                        )

    cli_args = parser.parse_args()

    # Open the config file
    with open(cli_args.config) as f:
        cfg = yaml.load(f, Loader=FullLoader)

    # Initiate model
    model = initModel(model_path=cfg["MODEL"])

    # Glob the audio files
    allFiles = [f for f in glob.glob(cfg["INPUT_PATH"] + "/**/*", recursive=True) if os.path.isfile(f)]
    print("Found {} files to analyze".format(len(allFiles)))

    # Make the predictions and write results
    for file_path in allFiles:
        analyzeFile(file_path, model, cfg["OUTPUT_PATH"])

# docker run --rm -it -v $pwd:/app/ -v ~/Data/:/Data registry.gitlab.com/nina-data/audioclip:latest poetry run python prediction_scripts/predict.py