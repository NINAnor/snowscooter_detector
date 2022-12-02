import numpy as np
import torch
import yaml
import itertools
import glob
import pandas as pd
from tqdm import tqdm

from torch.quantization import quantize_dynamic
from torch.utils.data import DataLoader, Dataset

from utils.utils_testing import AudioList, AudioLoader
from utils.parsing_utils import parseFolders

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix

import itertools

from model.custom_model import CustomAudioCLIP

identifier_string = "ben_ray_"
# model_path = "/home/femkegb/Documents/snowscooter_detector/experiment_output/tagged/grid_search_20221110_121145_wide_search_better_checkpointing/archive/20221111_002709/models/ckpt-epoch=47-val_loss=0.06.ckpt"

label_path = "/Data/test_dataset/labels"
audio_test_path = "/Data/test_dataset/audio"
# mpath = "/home/femkegb/Documents/snowscooter_detector/experiment_output/tagged/grid_search_20221115_095020_try_again_resume_dl2/archive/20221115_202556/models/ckpt-epoch=59-val_loss=0.07.ckpt" # femke model
# mpath = "/home/femkegb/Documents/snowscooter_detector/training/lightning_training/lightning_logs/ckpt-epoch=45-val_loss=0.09.ckpt" # ben params
# mpath = "/home/femkegb/Documents/snowscooter_detector/training/lightning_training/lightning_logs/ckpt-epoch=21-val_loss=0.12-lr=0.005.ckpt" # ben model
mpath = "/app/assets/ckpt-epoch=10-val_loss=0.09.ckpt"  # ben model ray


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

    return np.array(proba_list)


def initModel(model_path):
    model = CustomAudioCLIP(num_target_classes=2)
    model = model.load_from_checkpoint(model_path, num_target_classes=2)
    model.eval()

    # m = torch.load(model_path).eval()
    # m_q = quantize_dynamic(m, qconfig_spec={torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)
    return model


def get_preds_and_gt(item):

    gt = []  # ground truths
    preds = []  # predictions

    audio = item["audio"]
    result = item["result"]

    # Get the prediction scores
    l_audio = AudioList().get_processed_list(audio)
    test_loader = getTestLoader(l_audio[0])
    p = predict(test_loader, m)
    preds.append(p)

    # Get the ground truth
    df = pd.read_csv(result, sep="\t")
    df = df[df["View"] == "Spectrogram 1"].reset_index()

    for segment in range(len(l_audio[0])):

        if segment > 0:
            segment = segment * (l_segments - overlap)

        is_in_df = False

        for row in range(len(df)):

            begin_time = df.loc[row]["Begin Time (s)"]
            end_time = df.loc[row]["End Time (s)"]

            if begin_time <= segment <= end_time:
                is_in_df = True
            else:
                continue

        if is_in_df:
            gt.append(1)
        else:
            gt.append(0)

    return (gt, preds)


def get_auc_roc_fig(gt_array, preds_arr, title):

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
    plt.title(title)
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(identifier_string + "auc_roc.png")


def binarise(preds_array, threshold):

    predicted_labels = []

    for proba in preds_array:
        if proba > threshold:
            predicted_labels.append(1)
        else:
            predicted_labels.append(0)

    return predicted_labels


def get_confusion_matrix(gt, preds, threshold=0.9):

    b_preds = binarise(preds, threshold)
    print(confusion_matrix(gt, b_preds))


l_segments = 3
overlap = 0
time_index = 0
threshold = 0.9

flist = parseFolders(audio_test_path, label_path)
m = initModel(mpath)
gt_all = []  # ground truths
preds_all = []  # predictions
fp_info_all = []
fn_info_all = []

for item in tqdm(flist):
    gt, preds = get_preds_and_gt(item)
    gt_all.append(gt)
    preds_all.append(preds)
    # get audio file and segment number for False Positive
    detector_positives = preds[0] > 0.9
    ground_truth_positives = np.asarray(gt, dtype=bool)
    false_positives = np.logical_and(detector_positives, ~ground_truth_positives)
    false_positive_segments_start_time = np.where(false_positives)[0] * (
        l_segments - overlap
    )
    false_positive_segments_filename = [item["audio"]] * len(
        false_positive_segments_start_time
    )
    fp_info_all.extend(
        list(
            zip(
                false_positive_segments_filename,
                false_positive_segments_start_time.tolist(),
            )
        )
    )
    # get audio file and segment number for False Negative
    false_negatives = np.logical_and(~detector_positives, ground_truth_positives)
    false_negative_segments_start_time = np.where(false_negatives)[0] * (
        l_segments - overlap
    )
    false_negative_segments_filename = [item["audio"]] * len(
        false_negative_segments_start_time
    )
    fn_info_all.extend(
        list(
            zip(
                false_negative_segments_filename,
                false_negative_segments_start_time.tolist(),
            )
        )
    )
gt_all_all = list(itertools.chain.from_iterable(gt_all))
preds_all_all = list(itertools.chain.from_iterable(preds_all))

print("False negatives:")
print(fn_info_all)

print("False positives:")
print(fp_info_all)

get_auc_roc_fig(gt_all_all, np.concatenate(preds_all_all), title=identifier_string)
get_confusion_matrix(gt_all_all, np.concatenate(preds_all_all), threshold=0.9)

get_confusion_matrix(gt_all[26], preds_all[26][0])
get_confusion_matrix(gt_all[1], preds_all[1][0])

import csv

with open(identifier_string + "false_negatives.csv", "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow(["name", "t_start"])
    for row in fn_info_all:
        csv_out.writerow(row)

with open(identifier_string + "false_positives.csv", "w") as out:
    csv_out = csv.writer(out)
    csv_out.writerow(["name", "t_start"])
    for row in fp_info_all:
        csv_out.writerow(row)

import pickle

with open(identifier_string + "gt_all_all.pickle", "wb") as handle:
    pickle.dump(gt_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(identifier_string + "preds_all_all.pickle", "wb") as handle:
    pickle.dump(preds_all_all, handle, protocol=pickle.HIGHEST_PROTOCOL)
