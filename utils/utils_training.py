import torchvision as tv
import random

from utils.audio_processing import openAudioFile, splitSignal
from audiomentations import Compose, SevenBandParametricEQ, TimeMask, FrequencyMask, Shift, AirAbsorption, AddGaussianNoise

def transform_specifications(cfg):

    audio_transforms = Compose([
        AddGaussianNoise(min_amplitude=cfg['GAUSSIAN_MIN_AMPLITUDE'], max_amplitude=cfg['GAUSSIAN_MIN_AMPLITUDE'], p=cfg['GAUSSIAN_P']),
        SevenBandParametricEQ(p=cfg['P_SEVENBANDPARAMETRICEQ']),
        Shift(cfg['P_SHIFT']),
        #AirAbsorption(cfg['P_AIR_ABSORPTION']),
        TimeMask(cfg['P_TIME_MASK']),
        FrequencyMask(cfg['P_FREQ_MASK'])
        ]
    )
    return audio_transforms

class AudioList():

    def __init__(self, length_segments = 3, sample_rate=44100):
        self.sample_rate = sample_rate
        self.length_segments = length_segments

    def read_audio(self, audio_path):
        """Read the audio, change the sample rate and randomly pick one channel"""
        sig, _ = openAudioFile(audio_path, sample_rate=self.sample_rate)
        return sig

    def split_segment(self, array):
        splitted_array = splitSignal(array, rate=self.sample_rate, seconds=self.length_segments, overlap=0, minlen=3)
        return splitted_array

    def get_labels(self, splitted_list, label):
        arrays_label = []
        for array in splitted_list:
            array_label = (array, label)
            arrays_label.append(array_label)
        return arrays_label

    def get_processed_list(self, audio_path):

        list_segments = []

        for item in audio_path:
            track = self.read_audio(item)        
            label = item.split("/")[-2]
            list_divided = self.split_segment(track)
            list_arr_label = self.get_labels(list_divided, label)
            list_segments.append(list_arr_label)
        return list_segments