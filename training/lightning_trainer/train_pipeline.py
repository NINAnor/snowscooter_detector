#!/bin/usr/env python

import warnings
warnings.filterwarnings("ignore")

import torch
torch.cuda.empty_cache()

import glob
import random
import argparse
import pytorch_lightning as pl
import yaml
import os
import mlflow.pytorch
import json
import logging
import fs

import itertools

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.suggest.bayesopt import BayesOptSearch

from yaml.loader import FullLoader
from sklearn.preprocessing import LabelEncoder

from training.lightning_trainer.datamodule import EncodeLabels
from training.lightning_trainer.datamodule import AudioDataModule
from training.lightning_trainer.trainingmodule import TransferTrainingModule

from utils.utils_training import transform_specifications
from utils.utils_training import AudioList

class EncodeFileLabel():
    """
    Function that encodes names of folders as numerical labels
    Wrapper around sklearn's LabelEncoder
    """
    def __init__(self, label_file_path):
        self.label_file_path = label_file_path
        self.class_encode = LabelEncoder()
        self._labels_name()

    def _labels_name(self):

        with open(self.label_file_path) as f:
            label_file= json.load(f)

        dirs = []
        for item in label_file:
            dirs.append(item['Folder'])
        self.class_encode.fit(dirs)
        
    def __getLabels__(self):
        return self.class_encode.classes_

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]

    def one_hot_sample(self, label):
        t_label = self.to_one_hot(self.class_encode, [label])
        return t_label
        

def doConnection(connection_string):

    if connection_string is False:
        myfs = False
    else:
        myfs = fs.open_fs(connection_string)
    return myfs

def walk_audio(filesystem, input_path):
    
    walker = filesystem.walk(input_path, filter=['*.wav', '*.flac', '*.mp3', '*.ogg', '*.m4a', '*.WAV', '*.MP3'])
    for path, dirs, flist in walker:
        for f in flist:
            yield fs.path.combine(path, f.name)

def parseInputFiles(filesystem, input_path):

    if filesystem is False:
        files = [f for f in glob.glob(input_path + "/**/*", recursive=True) if os.path.isfile(f)]
        files = [f for f in files if f.endswith( (".WAV", ".wav", ".mp3") )]
    else:
        files = []
        for index, audiofile in enumerate(walk_audio(filesystem, input_path)):
            files.append(audiofile)
            
    print('Found {} files for training'.format(len(files)))

    return files

def getTrainValFilesLists(config, myfs):

    train_val_files = parseInputFiles(myfs, config["PATH_TRAIN_VAL_DATASET"])  

    # Split allFiles into a training / validation split
    train_samples = random.sample(train_val_files, int(len(train_val_files) * config["PROP_TRAINING"]))
    val_samples = [item for item in train_val_files if item not in train_samples]

    # Instantiate the audio iterator class - cut the audio into segments
    audio_list= AudioList(length_segments=config["LENGTH_SEGMENTS"], sample_rate=config["SAMPLE_RATE"], filesystem=myfs)

    list_train = audio_list.get_processed_list(train_samples)
    list_train = list(itertools.chain.from_iterable(list_train))

    list_val = audio_list.get_processed_list(val_samples)
    list_val = list(itertools.chain.from_iterable(list_val))

    return list_train, list_val

def getLabelEncoder(config, myfs):

    label_encoder = EncodeLabels(path_to_folders=config["PATH_TRAIN_VAL_DATASET"], filesystem=myfs)

    # Save name of the folder and associated label in a json file
    l = label_encoder.__getLabels__()
    t = label_encoder.class_encode.transform(l)

    folder_labels = []
    for i, j in zip(l,t):
        item = {"Folder": i, "Label": int(j)}
        folder_labels.append(item)

    if not os.path.isfile(config["LABEL_FILE"]):
        with open(config["LABEL_FILE"], 'w') as outfile:
            json.dump(folder_labels, outfile)

    return label_encoder

def callbacks(config):

    early_stopping = EarlyStopping(monitor="val_loss", patience=config["STOPPING_RULE_PATIENCE"])

    tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mean_accuracy": "val_accuracy"
    },
    on="validation_end")

    checkpoints_callback = ModelCheckpoint(
        dirpath=config["PATH_LIGHTNING_METRICS"],
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}")

    return [early_stopping, tune_callback, checkpoints_callback]

@mlflow_mixin
def run(config, list_train, list_val, callbacks):

    # Label encoder
    label_encoder = EncodeFileLabel(config["LABEL_FILE"])

    transform = transform_specifications(config)

    trainLoader, valLoader = AudioDataModule(list_train, list_val, label_encoder, 
                                batch_size=config["BATCH_SIZE"],
                                num_workers=config["NUM_WORKERS"],
                                pin_memory=config["PIN_MEMORY"],
                                transform=transform,
                                sampler=True # Should we oversample the training set?
                                ).train_val_loader()


    # Customize the training (add GPUs, callbacks ...)
    trainer = pl.Trainer(default_root_dir=config["PATH_LIGHTNING_METRICS"], 
                        max_epochs=config["N_EPOCHS"],
                        callbacks=callbacks,
                        accelerator=config["ACCELERATOR"]) 
    #trainer.save_checkpoint("example.ckpt")

    # Parameters for the training loop
    model_arguments = {
        "hop_length": config["FFT_HOP_LENGTH"],
        "n_fft": config["FFT_N_FFT"],
        "win_length": config["FFT_WIN_LENGTH"],
        "window": config["FFT_WINDOW"],
    }
    training_loop = TransferTrainingModule(learning_rate=config["LEARNING_RATE"], num_target_classes=config["NUM_TARGET_CLASSES"], model_arguments=model_arguments)

    # Finally train the model
    #with mlflow.start_run(experiment_id=config["current_experiment"]) as run:
    mlflow.pytorch.autolog(log_models = True)
    trainer.fit(training_loop, trainLoader, valLoader) 


@ray.remote(num_gpus=0.5)
def grid_search(config, list_train, list_val, cbacks):

    IP_HEAD_NODE = os.environ.get("IP_HEAD")
    print("HEAD NODE IP: {}".format(IP_HEAD_NODE))

    print("Ask for worker")

    # To run locally
    if IP_HEAD_NODE == None:
        options = {
            "object_store_memory": 10**9,
            "_temp_dir": "/rds/general/user/ss7412/home/AudioCLIP/",
            "address":"auto",
            "ignore_reinit_error":True
        }
    # To run one HPC
    else: 
        options = {
            "object_store_memory": 10**9,
            "_temp_dir":  "/rds/general/user/ss7412/home/AudioCLIP/",
            "_node_ip_address": IP_HEAD_NODE, 
            "address": os.environ.get("IP_HEAD_NODE"),
            "ignore_reinit_error":True
        }

    ray.init(**options)

    # For stopping non promising trials early
    scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=2)

    # Bayesian optimisation to sample hyperparameters in a smarter way
    algo = BayesOptSearch(random_search_steps=config["RANDOM_SEARCH_STEPS"], mode="min")

    reporter = CLIReporter(
        parameter_columns=["LEARNING_RATE", "BATCH_SIZE"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    resources_per_trial = {"cpu": config["N_CPU_PER_TRIAL"], "gpu": config["N_GPU_PER_TRIAL"]}

    trainable = tune.with_parameters(run, list_train=list_train, list_val=list_val, callbacks=cbacks)

    print("Running the trials")
    analysis = tune.run(trainable,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=config["N_SAMPLING"], # Number of times to sample from the hyperparameter space
        scheduler=scheduler,
        progress_reporter=reporter,
        name=config["NAME_EXPERIMENT"],
        local_dir=config["LOCAL_DIR"],
        search_alg=algo)

    print("Best hyperparameters found were: ", analysis.best_config)

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    print(df)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help="Path to the config file",
                        required=False,
                        default="./config_training.yaml",
                        type=str
    )
    
    parser.add_argument("--grid_search",
                        help="If grid search = True, the model will look for the best hyperparameters, else it will train",
                        required=False,
                        default=False,
                        type=str
    )

    cli_args = parser.parse_args()

    with open(cli_args.config) as f:
        config = yaml.load(f, Loader=FullLoader)

    mlflow.create_experiment(config["mlflow"]["experiment_name"])
    config["mlflow"]["tracking_uri"] = eval(config["mlflow"]["tracking_uri"])

    # Set the MLflow experiment, or create it if it does not exist.
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    # Do the connection
    myfs = doConnection(config["CONNECTION_STRING"])

    # Write the labels
    getLabelEncoder(config, myfs)

    # Get the train & val file list
    train_list, val_list = getTrainValFilesLists(config, myfs)

    # Get the list of callbacks
    cbacks = callbacks(config)

    # Run the script, with Ray.tune or not
    if cli_args.grid_search == "True":
        print("Begin the parameter search")
        for key in config.keys():
            try:
                config[key] = eval(config[key])
            except:
                continue
        results = grid_search.remote(config, train_list, val_list, cbacks)
        assert ray.get(results) == 1
    else:
        print("Begin the training script")
        run(config, train_list, val_list, cbacks)

# docker run --rm -it -v ~/Code/AudioCLIP:/app -v ~/Data/:/Data --gpus all audioclip:latest poetry run python lightning_trainer/train_pipeline.py
