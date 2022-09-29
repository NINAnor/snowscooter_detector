# :snowflake: Bioacoustic project using AudioCLIP

## Usage

Please make sure that the correct parameters are set on the `config_training.yaml` if you are going to train a model or `config_inference` if you are planning to use the model to detect snowscooters.

:star: Please note our scripts support [PyfileSystem](https://www.pyfilesystem.org/). This means that it is possible to add a `CONNECTION_STRING` argument in the form `user:password@host`  in `config_inference` or `config_training` for the scripts to fetch the data on your remote server.

:star: **If your data is stored locally** simply set `CONNECTION_STRING` to `False`.

### Installation using Docker

1. Clone the repository:

```
git clone https://github.com/NINAnor/snowscooter_detector
```

2. Build the image


```
cd AudioCLIP
docker build -t audioclip -f Dockerfile .
```

Build the image with Singularity (if you are working on HPC)

```
cd AudioCLIP
singularity build
```

### Train model

To train a model, simply update the path and the parameters in the `config_training.yaml` and run `train.sh`.

As of now, `train.sh` can only be used with `docker` and you need to specify the path that contains your data (training data + short noises and background noises if you have any) so that it can be mounted in the container. The command should look like:

```
./train.sh /PATH/CONTAINING/ALL/DATA
```

### Predict on new files

```
OUT_FOLDER=results
mkdir -p $OUT_FOLDER

docker run --gpus all --v $PWD:/app audioclip poetry run python /app/inference/predict.py
```

## Acknowledgment

This project is the result of a collaboration between [NINA](https://www.nina.no/english) and [SINTEF](https://www.sintef.no/en/) and has been funded by the Norwegian Research Council




