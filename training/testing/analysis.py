import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../.."))
from copy import deepcopy

import numpy as np
import pandas as pd
import csv
import librosa
import soundfile as sf
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

from utils.parsing_utils import remove_extension
from test import L_SEGMENTS

HR_THRESHOLD = 0.05
DETECTOR_THRESHOLD = 0.95
MIN_N_SEGMENTS_FOR_DETECTION = 2
SAVE_SEGMENTS_TO_WAV = False


def calc_performance(gt_all_all, preds_all_all, hr_all_all):
    """Converts results of test.py into intervals and obtains performance scores"""
    # Obtain ground truth intervals
    gt_intervals = np.flatnonzero(
        np.diff(gt_all_all, prepend=False, append=False)
    ).reshape(-1, 2) - (0, 1)
    gt_intervals_pd = pd.arrays.IntervalArray.from_arrays(
        gt_intervals[:, 0], gt_intervals[:, 1], closed="both"
    )

    # Find where hr is too low (= probably wind) and replace predictions
    hr_negatives = hr_all_all < HR_THRESHOLD
    preds_new = deepcopy(preds_all_all)
    preds_new[np.logical_and(preds_all_all > DETECTOR_THRESHOLD, hr_negatives)] = 0

    # Obtain prediction intervals
    pred_intervals = np.flatnonzero(
        np.diff(preds_new > DETECTOR_THRESHOLD, prepend=False, append=False)
    ).reshape(-1, 2) - (0, 1)
    pred_intervals_pd = pd.arrays.IntervalArray.from_arrays(
        pred_intervals[:, 0], pred_intervals[:, 1], closed="both"
    )

    # Remove prediction intervals that are too short in duration
    pred_intervals_pd = pred_intervals_pd[
        pred_intervals_pd.length > MIN_N_SEGMENTS_FOR_DETECTION
    ]

    # Obtain results for FP/FN/TP
    gt_interval_is_in_pred = np.full((len(gt_intervals_pd)), False)
    pred_interval_is_in_gt = np.full((len(pred_intervals_pd)), False)
    for gt_ind, gt_interval in enumerate(gt_intervals_pd):
        for pred_ind, pred_interval in enumerate(pred_intervals_pd):
            if gt_interval.overlaps(pred_interval):
                pred_interval_is_in_gt[pred_ind] = True
                gt_interval_is_in_pred[gt_ind] = True
                # Note: do not break, there may be multiple overlaps

    FP = np.sum(~pred_interval_is_in_gt)  # all pred intervals not in gt intervals
    TP = np.sum(gt_interval_is_in_pred)  # all gt intervals also in pred intervals
    FN = np.sum(~gt_interval_is_in_pred)  # all gt intervals not in pred intervals

    # Calculate scores
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1_score = 2 * precision * recall / (precision + recall)

    # print results
    print("HR threshold: " + str(HR_THRESHOLD))
    print("Detector threshold: " + str(DETECTOR_THRESHOLD))
    print("F1-score: " + str(F1_score))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("FN: " + str(FN))
    print("FP: " + str(FP))
    print("TP: " + str(TP))

    return (
        pred_intervals_pd,
        pred_interval_is_in_gt,
        gt_intervals_pd,
        gt_interval_is_in_pred,
    )


def save_detection_details(
    pd_intervals_pd,
    results_pd_intervals,
    gt_intervals_pd,
    results_gt_interval,
    result_path,
    segment_ids_all,
    gt_all_all,
    preds_all_all,
    hr_all_all,
    comments_all_all,
    subfolder_name,
):
    """Save all FP/TP/FN intervaøs and the raw data to separate csv files"""
    results_path_extended = os.path.join(result_path, subfolder_name)
    if not os.path.exists(results_path_extended):
        os.makedirs(results_path_extended)
    if SAVE_SEGMENTS_TO_WAV:
        for wav_dir in ["FN", "TP", "FP"]:
            if not os.path.exists(os.path.join(results_path_extended, "wavs", wav_dir)):
                os.makedirs(os.path.join(results_path_extended, "wavs", wav_dir))

    with open(os.path.join(results_path_extended, "FP_intervals.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(
            ["name", "t_start", "n_segments", "comment", "mean hr", "mean output"]
        )
        for ind, row in enumerate(pd_intervals_pd):
            if not results_pd_intervals[ind]:
                if SAVE_SEGMENTS_TO_WAV:
                    save_segment_to_wav(
                        segment_ids_all,
                        row,
                        os.path.join(results_path_extended, "wavs", "FP"),
                    )
                csv_out.writerow(
                    (
                        str(segment_ids_all[row.left][0]),
                        segment_ids_all[row.left][1],
                        row.right - row.left,
                        comments_all_all[row.left + (row.right - row.left) // 2],
                        np.mean(hr_all_all[row.left : row.right]),
                        np.mean(preds_all_all[row.left : row.right]),
                    )
                )
    with open(os.path.join(results_path_extended, "FN_intervals.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(
            ["name", "t_start", "n_segments", "comment", "mean hr", "mean output"]
        )
        for ind, row in enumerate(gt_intervals_pd):
            if not results_gt_interval[ind]:
                if SAVE_SEGMENTS_TO_WAV:
                    save_segment_to_wav(
                        segment_ids_all,
                        row,
                        os.path.join(results_path_extended, "wavs", "FN"),
                    )
                csv_out.writerow(
                    (
                        str(segment_ids_all[row.left][0]),
                        segment_ids_all[row.left][1],
                        row.right - row.left,
                        comments_all_all[row.left + (row.right - row.left) // 2],
                        np.mean(hr_all_all[row.left : row.right]),
                        np.mean(preds_all_all[row.left : row.right]),
                    )
                )

    with open(os.path.join(results_path_extended, "TP_intervals.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["name", "t_start", "n_segments"])
        for ind, row in enumerate(gt_intervals_pd):
            if results_gt_interval[ind]:
                if SAVE_SEGMENTS_TO_WAV:
                    save_segment_to_wav(
                        segment_ids_all,
                        row,
                        os.path.join(results_path_extended, "wavs", "TP"),
                    )
                csv_out.writerow(
                    (
                        str(segment_ids_all[row.left][0]),
                        segment_ids_all[row.left][1],
                        row.right - row.left,
                        comments_all_all[row.left + (row.right - row.left) // 2],
                        np.mean(hr_all_all[row.left : row.right]),
                        np.mean(preds_all_all[row.left : row.right]),
                    )
                )
    with open(os.path.join(results_path_extended, "raw_data.csv"), "w") as out:
        csv_out = csv.writer(out)
        csv_out.writerow(["name", "t_start", "comment", "mean hr", "mean output", "gt"])
        for ind, row in enumerate(gt_all_all):
            csv_out.writerow(
                (
                    str(segment_ids_all[ind][0]),
                    segment_ids_all[ind][1],
                    comments_all_all[ind],
                    hr_all_all[ind],
                    preds_all_all[ind],
                    gt_all_all[ind],
                )
            )


def save_segment_to_wav(segment_ids_all, row, folder_path):
    """Save the identified segment to a wav file"""
    sig, rate = librosa.load(
        str(segment_ids_all[row.left][0]),
        offset=float(segment_ids_all[row.left][1]),
        duration=(row.right - row.left) * 3,
        mono=True,
        res_type="kaiser_fast",
    )
    sf.write(
        os.path.join(
            folder_path,
            remove_extension(str(segment_ids_all[row.left][0]))
            + "_"
            + segment_ids_all[row.left][1]
            + ".wav",
        ),
        sig,
        rate,
        "PCM_16",
    )


def run_analysis(results_path):
    """Summarize the results in various ways"""
    ## obtain earlier saved test results
    gt_all_all = np.load(os.path.join(results_path, "gt_all_all.npy"))
    preds_all_all = np.load(os.path.join(results_path, "preds_all_all.npy"))
    hr_all_all = np.load(os.path.join(results_path, "hr_all_all.npy"))
    segment_ids_all = np.load(os.path.join(results_path, "segment_ids_all.npy"))
    comments_all_all = np.load(os.path.join(results_path, "comments_all_all.npy"))

    ## Obtain scores when including all engines as true positives
    print("Analysis including all engines")
    # Set gt for wind labels to zero
    ind_to_nullify_gt = np.full((len(comments_all_all)), False)
    for ind, comment in enumerate(comments_all_all):
        if "wind" in comment:
            ind_to_nullify_gt[ind] = True
    gt_all_engines = deepcopy(gt_all_all)
    gt_all_engines[ind_to_nullify_gt] = 0
    # Obtain scores
    (
        pred_intervals_pd,
        pred_interval_is_in_gt,
        gt_intervals_pd,
        gt_interval_is_in_pred,
    ) = calc_performance(gt_all_engines, preds_all_all, hr_all_all)

    # Save detection details into csv files
    save_detection_details(
        pred_intervals_pd,
        pred_interval_is_in_gt,
        gt_intervals_pd,
        gt_interval_is_in_pred,
        results_path,
        segment_ids_all,
        gt_all_all,
        preds_all_all,
        hr_all_all,
        comments_all_all,
        "all_engines",
    )

    ## Obtain scores when including only snowmobiles as true positives
    print("including only snowmobiles")
    # Only keep gt of sw and None labelss, set rest to zero
    ind_to_nullify_gt = np.full((len(comments_all_all)), True)
    for ind, comment in enumerate(comments_all_all):
        if "sw" in comment or "None" in comment:
            ind_to_nullify_gt[ind] = False
    gt_sw_only = deepcopy(gt_all_all)
    gt_sw_only[ind_to_nullify_gt] = 0
    # Obtain scores
    (
        pred_intervals_pd,
        pred_interval_is_in_gt,
        gt_intervals_pd,
        gt_interval_is_in_pred,
    ) = calc_performance(gt_sw_only, preds_all_all, hr_all_all)

    # Save detection details into csv files
    save_detection_details(
        pred_intervals_pd,
        pred_interval_is_in_gt,
        gt_intervals_pd,
        gt_interval_is_in_pred,
        results_path,
        segment_ids_all,
        gt_all_all,
        preds_all_all,
        hr_all_all,
        comments_all_all,
        "only_sw",
    )

    # Plot precision vs recall figure for sw only NOTE: on segment data, very biased
    plt.figure()
    for hr_threshold in np.linspace(0.01, 0.05, 6):
        hr_negatives = (
            hr_all_all < hr_threshold
        )  # find where hr is too low (not machine made noise)
        preds_new = deepcopy(preds_all_all)
        preds_new[np.logical_and(preds_all_all > DETECTOR_THRESHOLD, hr_negatives)] = 0
        precision, recall, thresholds = precision_recall_curve(gt_all_all, preds_new)
        ap = average_precision_score(gt_all_all, preds_new, average="weighted")
        plt.plot(
            recall,
            precision,
            label="hr_threshold="
            + str(hr_threshold.round(2))
            + " with ap="
            + str(ap.round(2)),
        )
    plt.legend(loc="center left")
    plt.ylabel("precision")
    plt.xlabel("recall")
    plt.savefig(os.path.join(results_path, "only_sw", "precision_recall.png"))

    # Calculate time-based confusion matrix
    FN_intervals = gt_intervals_pd[~gt_interval_is_in_pred]
    total_time_FN = (
        np.sum((FN_intervals.right - FN_intervals.left + 1)) * L_SEGMENTS / 60
    )
    TP_intervals = gt_intervals_pd[gt_interval_is_in_pred]
    total_time_TP = (
        np.sum((TP_intervals.right - TP_intervals.left + 1)) * L_SEGMENTS / 60
    )
    FP_intervals = pred_intervals_pd[~pred_interval_is_in_gt]
    total_time_FP = (
        np.sum((FP_intervals.right - FP_intervals.left + 1)) * L_SEGMENTS / 60
    )
    total_time_TN = (
        len(gt_all_all) * L_SEGMENTS / 60
        - total_time_TP
        - total_time_FP
        - total_time_FN
    )
    total_time = len(gt_all_all) * L_SEGMENTS / 60
    print(
        "TP: "
        + str(total_time_TP)
        + " min ("
        + str(total_time_TP / total_time * 100)
        + "%)"
    )
    print(
        "FP: "
        + str(total_time_FP)
        + " min ("
        + str(total_time_FP / total_time * 100)
        + "%)"
    )
    print(
        "FN: "
        + str(total_time_FN)
        + " min ("
        + str(total_time_FN / total_time * 100)
        + "%)"
    )
    print(
        "TN: "
        + str(total_time_TN / 60)
        + " hours ("
        + str(total_time_TN / total_time * 100)
        + "%)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results_path",
        help="Path to the directory where subdirectory with results will be created/overwritten",
        required=False,
        default="results/",
        type=str,
    )

    parser.add_argument(
        "--model_path",
        help="Path to the model weights",
        required=False,
        default="/app/assets/ckpt-epoch=21-val_loss=0.12-lr=0.005.ckpt",
        type=str,
    )

    cli_args = parser.parse_args()

    # find subfolder for this specific model in the results directory
    model_identifier = remove_extension(cli_args.model_path)
    results_path = os.path.join(cli_args.results_path, model_identifier)
    run_analysis(results_path)
