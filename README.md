# :snowflake: Bioacoustic project using AudioCLIP

## Usage

Please make sure that the correct parameters are set on the `config_training.yaml` if you are going to train a model or `config_inference` if you are planning to use the model to detect snowscooters.

### Installation using Docker

1. Clone the repository:

``````
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


### Predict on new files

1. PyfileSystem

Please note that we are using [PyfileSystem](https://www.pyfilesystem.org/)  in `inference/predict.py` to make the scripts agnostic to files location. This means that in the `config` files you will find the parameters `CONNECTION_STRING` and `INPUT_PATH`.

If the training data or the data you want to be predicted are located on a remote server the `CONNECTION_STRING` should contain the connection protocol, user, password and host and should be in the form:

```
ssh://[user[:password]@]host[:port]
```

If the data is stored locally, simply set `CONNECTION_STRING` to `None`

2. Running the prediction script

```
OUT_FOLDER=results
mkdir -p $OUT_FOLDER

docker run --gpus all --v $PWD:/app audioclip poetry run python /app/inference/predict.py
```

## Acknowledgment

This project is the result of a collaboration between [NINA](https://www.nina.no/english) and [SINTEF](https://www.sintef.no/en/) and has been funded by the Norwegian Research Council




