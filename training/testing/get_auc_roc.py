import argparse
import yaml
import numpy as np
import torch
import yaml
import itertools
import glob
import pandas as pd
import matplotlib.pyplot as plt

from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from utils.utils_testing import AudioList, AudioLoader, parseFolders


def getTestLoader(list_arrays):
    list_preds = AudioLoader(list_arrays)
    predLoader = DataLoader(list_preds, batch_size=1, num_workers=4, pin_memory=False)
    return predLoader

def predict(testLoader, model):

    proba_list = []

    for array in testLoader:
        tensor = array
        output = model(tensor)
        output = np.exp(output.detach().numpy())
        proba_list.append(output[0][1])

    return (np.array(proba_list))

def initModel(model_path):
    m = torch.load(model_path).eval()
    m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return m_q

def get_preds_and_gt(item, model, l_segments, overlap):

    gt = []
    preds = []

    audio = item['audio']
    result = item['result']

    # Get the prediction scores
    l_audio = AudioList().get_processed_list(audio)
    test_loader = getTestLoader(l_audio[0])
    p = predict(test_loader, model)
    preds.append(p)

    # Get the ground truth
    df = pd.read_csv(result, sep='\t')
    df = df[df['View'] == 'Spectrogram 1'].reset_index()

    for segment in range(len(l_audio[0])):

        if segment > 0:
            segment = segment * (l_segments - overlap)
            
        is_in_df = False

        for row in range(len(df)):

            begin_time = df.loc[row]['Begin Time (s)']
            end_time = df.loc[row]['End Time (s)']

            if begin_time <= segment <= end_time:
                is_in_df = True
            else:
                continue
        
        if is_in_df:
            gt.append(1)
        else: 
            gt.append(0)

    return (gt, preds)

def get_auc_roc_fig(gt_array, preds_arr):
    
    fpr, tpr, _ = roc_curve(gt_array, preds_arr)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic example")
    plt.legend(loc="lower right")
    plt.savefig('auc_roc.png')

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

    flist = parseFolders(cfg['TEST_PATH'], cfg['TEST_DF_PATH'])

    m = initModel(cfg["MODEL"])

    gt_all = []
    preds_all = []

    for item in flist:
        gt, preds = get_preds_and_gt(item)
        gt_all.append(gt)
        preds_all.append(preds)

    get_auc_roc_fig(np.array(gt), np.array(preds[0]))

