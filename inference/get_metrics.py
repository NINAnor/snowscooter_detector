import argparse
import numpy as np
import torch
import yaml
import itertools
import glob
import os
import pandas as pd
import matplotlib.pyplot as plt

from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

from prediction_scripts._utils import AudioListMetrics, AudioLoader, EncodeLabels

def initModel(model_path):
    m = torch.load(model_path).eval()
    m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return m_q
    
def getPredLoader(list_arrays, l):
    list_preds = AudioLoader(list_arrays, label_encoder)
    predLoader = DataLoader(list_preds, batch_size=1, num_workers=4, pin_memory=False)
    return predLoader

def predict(testLoader, model):

    proba_list = []
    label_list = []

    for array, label in testLoader:
        tensor = torch.tensor(array)
        output = model(tensor)
        output = np.exp(output.detach().numpy())
        proba_list.append(output[0][1])
        label_list.append(label[0])

    return (np.array(proba_list), np.array(label_list))

def threshold_df(labels, proba_list):
    precision, recall, thresholds = precision_recall_curve(labels, proba_list)
    thresholds = np.append(thresholds, 1)

    d = {'thresholds': thresholds, 'precision': precision, 'recall': recall}
    df = pd.DataFrame(data=d)
    df.to_csv('thresholds_df.csv')

def auc_roc_curve(labels, proba_list):
    fpr, tpr, thresholds = roc_curve(labels, proba_list)
    auc = roc_auc_score(labels, proba_list)

    #create ROC curve
    plt.plot(fpr,tpr,label="AUC="+str(round(auc, 2)))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.savefig('auc_roc_curve.png')

if __name__ == "__main__":

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
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    ###################
    # Get the dataset #
    ###################
    allFiles = [f for f in glob.glob(cfg["INPUT_PATH"] + "/**/*", recursive=True) if os.path.isfile(f)]
    allFiles = [f for f in allFiles if f.endswith( (".WAV", ".wav", ".mp3") )]

    # Instantiate the audio iterator class - cut the audio into segments
    audio_list= AudioListMetrics(length_segments=cfg["SIG_LENGTH"], sample_rate=cfg["SAMPLE_RATE"])

    list_test = audio_list.get_processed_list(allFiles)
    list_test = list(itertools.chain.from_iterable(list_test))

    ###########################
    # Create the labelEncoder #
    ###########################
    label_encoder = EncodeLabels(path_to_folders=cfg["INPUT_PATH"])

    # Save name of the folder and associated label in a json file
    l = label_encoder.__getLabels__()
    t = label_encoder.class_encode.transform(l)

    folder_labels = []
    for i, j in zip(l,t):
        item = {"Folder": i, "Label": int(j)}
        folder_labels.append(item)

    ######################
    # Do the predictions #
    ######################
    audioloader = AudioLoader(list_test, label_encoder)
    predLoader = DataLoader(audioloader, batch_size=1, num_workers=4, pin_memory=False)
    model = initModel(cfg["MODEL"])
    proba_list, labels = predict(predLoader, model)

    ###########################################
    # Calculate precision, recall & threshold #
    ###########################################
    threshold_df(labels, proba_list)
    auc_roc_curve(labels, proba_list)

# docker run --rm -it -v $pwd:/app/ -v ~/Data/:/Data registry.gitlab.com/nina-data/audioclip:latest poetry run python prediction_scripts/get_metrics.py