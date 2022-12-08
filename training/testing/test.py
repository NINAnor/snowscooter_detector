import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
import itertools


import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.utils_testing import AudioList, AudioLoader
from utils.parsing_utils import parseFolders
from utils.audio_signal import AudioSignal
from model.custom_model import CustomAudioCLIP


L_SEGMENTS = 3
OVERLAP = 0


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


def initModel(model_path, model_arguments):
    model = CustomAudioCLIP(num_target_classes=2, model_arguments=model_arguments)
    model = model.load_from_checkpoint(model_path, num_target_classes=2)
    model.eval()
    return model


def get_preds_and_gt(model, item, calculate_hr=False):
    gt = []  # ground truths
    preds = []  # predictions
    hr = []  # harmonic ratios
    comment = []  # tagging type / comments

    audio = item["audio"]
    result = item["result"]

    # Get the prediction scores
    l_audio = AudioList().get_processed_list(audio)
    test_loader = getTestLoader(l_audio[0])
    pred = predict(test_loader, model)
    preds.append(pred)

    # Get the ground truth
    df = pd.read_csv(result, sep="\t")

    for i in range(len(l_audio[0])):
        segment = i * (L_SEGMENTS - OVERLAP)
        df_filtered_on_segment = df[
            (segment <= df["End Time (s)"]) & (df["Begin Time (s)"] <= segment)
        ]
        is_in_df = len(df_filtered_on_segment) > 0

        if is_in_df:
            gt.append(1)
            comment.append(
                str(df_filtered_on_segment["type"].values[0])
                + " / "
                + str(df_filtered_on_segment["comment"].values[0])
            )
        else:
            gt.append(0)
            comment.append("None")
        if calculate_hr:
            if pred[i] > 0.0:
                signal = AudioSignal(l_audio[0][i], fs=44100)
                signal.apply_butterworth_filter(
                    order=18, Wn=np.asarray([1, 600]) / (signal.fs / 2)
                )
                signal_hr = signal.harmonic_ratio(
                    win_length=int(1 * signal.fs),
                    hop_length=int(0.1 * signal.fs),
                    window="hamming",
                )
                hr.append(np.mean(signal_hr))
            else:
                hr.append(-1)

    return (gt, preds, hr, comment)


def test_model(audio_test_path, label_path, mpath, model_arguments):
    # obtain audio and labels
    flist = parseFolders(audio_test_path, label_path)

    # initialize the model
    model = initModel(mpath, model_arguments=model_arguments)

    # init list for information tracking
    gt_all = []  # ground truths
    preds_all = []  # predictions
    segment_ids_all = []  # tuples of filename and segment start time
    hr_all = []  # harmonic ratios
    comments_all = []  # comments from labels (None if no comment)

    for item in tqdm(flist):
        # run model
        gt, preds, hr, comment = get_preds_and_gt(model, item, calculate_hr=True)

        # collect results
        gt_all.append(gt)
        preds_all.append(preds)
        hr_all.append(hr)
        comments_all.append(comment)

        # get segment ids for later identification
        segment_ids = [item["audio"]] * len(hr)
        segment_ids_start_time = np.linspace(
            0.0, len(hr) * (L_SEGMENTS - OVERLAP) - (L_SEGMENTS - OVERLAP), len(hr)
        )
        segment_ids_all.extend(list(zip(segment_ids, segment_ids_start_time.tolist())))

    gt_all_all = np.asarray(list(itertools.chain.from_iterable(gt_all)))
    preds_all_all = np.concatenate(list(itertools.chain.from_iterable(preds_all)))
    hr_all_all = np.asarray(list(itertools.chain.from_iterable(hr_all)))
    comments_all_all = np.asarray(list(itertools.chain.from_iterable(comments_all)))

    np.save(identifier_string + "gt_all_all.npy", gt_all_all)
    np.save(identifier_string + "preds_all_all.npy", preds_all_all)
    np.save(identifier_string + "hr_all_all.npy", hr_all_all)
    np.save(identifier_string + "comments_all_all.npy", comments_all_all)
    np.save(identifier_string + "segment_ids_all.npy", np.asarray(segment_ids_all))


if __name__ == "__main__":
    label_path = "/Data/test_dataset/labels"
    audio_test_path = "/Data/test_dataset/audio"
    mpath = "/app/assets/ckpt-epoch=21-val_loss=0.12-lr=0.005.ckpt"
    identifier_string = "ben_model_"
    model_arguments = {}

    test_model(audio_test_path, label_path, mpath, model_arguments)


    
