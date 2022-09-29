<h1 align="center"> :snowflake: SnowScooterDetector :snowflake: </h1>

![CC BY-NC-SA 4.0][license-badge]
![Supported OS][os-badge]

[license-badge]: https://badgen.net/badge/License/CC-BY-NC-SA%204.0/green
[os-badge]: https://badgen.net/badge/OS/Linux%2C%20Windows/blue


## Usage

Please make sure that the correct parameters are set on the `config_training.yaml` if you are going to train a model or `config_inference` if you are planning to use the model to detect snowscooters.

## Installation

### Installation without Docker

This code has been tested using Ubuntu 18.04 LTS and Windows 10 but should work with other distributions as well. Only Python 3.8 is supported though the code should work with other distributions as well.

1. Clone the repository:

`git clone https://github.com/NINAnor/snowscooter_detector`

2. Install requirements:

We use [poetry](https://python-poetry.org/) as a package manager which can be installed with the instructions below:

```
cd snowscooter_detector
pip install poetry 
poetry install --no-root
```

3. Pydub and Librosa require audio backend (FFMPEG)

`sudo apt-get install ffmpeg`

### Installation using Docker

First you need to have docker installed on your machine. Please follow the guidelines from the [official documentation](https://docs.docker.com/engine/install/).

1. Clone the repository:

`git clone https://github.com/NINAnor/snowscooter_detector`

2. Build the image

```
cd snowscooter_detector
docker build -t SnowscooterDet -f Dockerfile .
```

### Training your own model using our pipeline

After updating the parameters in `config_training.yaml` run the training script:

`poetry run python training/lightning_trainer/train_pipeline.py`

Or alternatively, if you have docker installed and the docker image built:

`docker run --rm -v $PWD/:/app -v /PATH/TO/YOUR/DATA/:/Data SnowscooterDet python training/lightning_trainer/train_pipeline.py`

:bulb: If you are using Docker, **don't forget** the flag `--nv` to expose your GPU :bulb:

### Predict on new files

After updating the parameters in `config_inference.yaml` run the prediction script:

`poetry run python inference/predict.py`

Or alternatively, if you have docker installed and the docker image built:

`docker run --rm -v $PWD/:/app -v /PATH/TO/YOUR/DATA/:/Data SnowscooterDet python inference/predict.py`

:bulb: If you are using Docker, **don't forget** the flag `--nv` to expose your GPU :bulb:

## PyfileSystem (optional)

:star: **Please note our scripts support [PyfileSystem](https://www.pyfilesystem.org/)**. This means that it is possible to add a `CONNECTION_STRING` argument in the form `user:password@host`  in `config_inference` or `config_training` for the scripts to fetch the data on your remote server.

:star: **If your data is stored locally** simply set `CONNECTION_STRING` to `False`.

## Contact

If you come across any issues with any of our scripts, please open an **issue** on the project's GitHub.

For other inquiry you can contact me at *benjamin.cretois@nina.no*.

## Acknowledgment

This project is the result of a collaboration between [NINA](https://www.nina.no/english) and [SINTEF](https://www.sintef.no/en/) and has been funded by the Norwegian Research Council.




