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

import itertools

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.loggers import TensorBoardLogger

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.integration.mlflow import mlflow_mixin
from ray.tune.suggest.bayesopt import BayesOptSearch

from yaml.loader import FullLoader

from lightning_trainer.datamodule import EncodeLabels
from lightning_trainer.datamodule import AudioDataModule
from lightning_trainer.trainingmodule import TransferTrainingModule

from lightning_trainer._utils import transform_specifications
from lightning_trainer._utils import AudioList

@mlflow_mixin
def run(config):
    #############################
    # Create the data iterators #
    #############################
    allFiles = [f for f in glob.glob(config["train_val_dataset_path"], recursive=True) if os.path.isfile(f)]
    allFiles = [f for f in allFiles if f.endswith( (".WAV", ".wav", ".mp3") )]

    # Split allFiles into a training / validation split
    train_samples = random.sample(allFiles, int(len(allFiles) * config["proportion_training"]))
    val_samples = [item for item in allFiles if item not in train_samples]

    # Instantiate the audio iterator class - cut the audio into segments
    audio_list= AudioList(length_segments=config["length_segments"], sample_rate=config["sample_rate"])

    list_train = audio_list.get_processed_list(train_samples)
    list_train = list(itertools.chain.from_iterable(list_train))

    list_val = audio_list.get_processed_list(val_samples)
    list_val = list(itertools.chain.from_iterable(list_val))

    ###########################
    # Create the labelEncoder #
    ###########################
    label_encoder = EncodeLabels(path_to_folders=config["train_val_dataset_path"])

    # Save name of the folder and associated label in a json file
    l = label_encoder.__getLabels__()
    t = label_encoder.class_encode.transform(l)

    folder_labels = []
    for i, j in zip(l,t):
        item = {"Folder": i, "Label": int(j)}
        folder_labels.append(item)

    if not os.path.isfile('/app/assets/label_correspondance.json'):
        with open('/app/assets/label_correspondance.json', 'w') as outfile:
            json.dump(folder_labels, outfile)

    ########################
    # Define the callbacks #
    ########################
    early_stopping = EarlyStopping(monitor="val_loss", patience=config["stopping_rule_patience"])

    tune_callback = TuneReportCallback(
    {
        "loss": "val_loss",
        "mean_accuracy": "val_accuracy"
    },
    on="validation_end")

    checkpoints_callback = ModelCheckpoint(
        dirpath=config["path_lightning_metrics"],
        monitor="val_loss",
        filename="ckpt-{epoch:02d}-{val_loss:.2f}")

    #########################################################################
    # Instantiate the trainLoader and valLoader and train the model with pl #
    #########################################################################
    transform = transform_specifications(config)

    trainLoader, valLoader = AudioDataModule(list_train, list_val, label_encoder, 
                                batch_size=config["batch_size"],
                                num_workers=config["num_workers"],
                                pin_memory=config["pin_memory"],
                                transform=transform,
                                sampler=True # Should we oversample the training set?
                                ).train_val_loader()


    # Customize the training (add GPUs, callbacks ...)
    trainer = pl.Trainer(default_root_dir=config["path_lightning_metrics"], 
                        #logger=TensorBoardLogger(save_dir=tune.get_trial_dir(), name="", version="."),
                        max_epochs=config["n_epoch"],
                        callbacks=[tune_callback, early_stopping, checkpoints_callback],
                        accelerator=config["accelerator"]) 
    #trainer.save_checkpoint("example.ckpt")

    # Parameters for the training loop
    training_loop = TransferTrainingModule(learning_rate=config["learning_rate"], num_target_classes=config["num_target_classes"])

    # Finally train the model
    #with mlflow.start_run(experiment_id=config["current_experiment"]) as run:
    mlflow.pytorch.autolog(log_models = True)
    trainer.fit(training_loop, trainLoader, valLoader) 

@ray.remote(num_gpus=1)
def grid_search(config):

    IP_HEAD_NODE = os.environ.get("IP_HEAD")
    print("HEAD NODE IP: {}".format(IP_HEAD_NODE))

    print("Ask for worker")
    options = {
        "object_store_memory": 10**9,
        "_temp_dir": "/rds/general/user/ss7412/home/AudioCLIP/",
        "_node_ip_address": IP_HEAD_NODE, #"address":"auto",
        "address": os.environ.get("IP_HEAD"),
        "ignore_reinit_error":True
    }
    #host = os.environ["RAY_HOST"]
    #if os.environ.get("RAY_MODE", "head") == "head":
    #     options.update(**{
    #        "_node_ip_address": host,
    #     })

    #else:
    #     options.update(**{
    #        "address": host
    #     })

    ray.init(**options)

    # For stopping non promising trials early
    scheduler = ASHAScheduler(
        max_t=5,
        grace_period=1,
        reduction_factor=2)

    # Bayesian optimisation to sample hyperparameters in a smarter way
    #algo = BayesOptSearch(random_search_steps=4, mode="min")

    reporter = CLIReporter(
        parameter_columns=["learning_rate", "batch_size"],
        metric_columns=["loss", "mean_accuracy", "training_iteration"])

    resources_per_trial = {"cpu": config["n_cpu_per_trials"], "gpu": config["n_gpu_per_trials"]}

    trainable = tune.with_parameters(run)

    print("Running the trials")
    analysis = tune.run(trainable,
        resources_per_trial=resources_per_trial,
        metric="loss",
        mode="min",
        config=config,
        num_samples=config["n_sampling"], # Number of times to sample from the hyperparameter space
        scheduler=scheduler,
        progress_reporter=reporter,
        name=config["name_experiment"],
        local_dir="/rds/general/user/ss7412/home/AudioCLIP/")
        #search_alg=algo)

    print("Best hyperparameters found were: ", analysis.best_config)

    # Get a dataframe for analyzing trial results.
    df = analysis.results_df
    print(df)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--config",
                        help="Path to the config file",
                        required=False,
                        default="/app/lightning_trainer/config.yaml",
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
    mlflow.set_tracking_uri(None)
    mlflow.set_experiment(config["mlflow"]["experiment_name"])


    if cli_args.grid_search == "True":
        #for key in ('learning_rate'): # , 'batch_size'
            #config[key] = eval(config[key])
        print("Begin the parameter search")
        config["learning_rate"] = eval(config["learning_rate"])
        results = grid_search.remote(config)
        assert ray.get(results) == 1
    else:
        print("Begin the training script")
        run(config)

# docker run --rm -it -v ~/Code/AudioCLIP:/app -v ~/Data/:/Data --gpus all audioclip:latest poetry run python lightning_trainer/train_pipeline.py
