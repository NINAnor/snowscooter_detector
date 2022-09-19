import torch
import numpy as np

from torch.utils.data import Dataset

from utils.audio_processing import openAudioFile, splitSignal, noise

RANDOM = np.random.RandomState(42)

class AudioList():

    def __init__(self, length_segments = 3, overlap = 0, sample_rate=44100):
        self.sample_rate = sample_rate
        self.length_segments = length_segments
        self.overlap = overlap
        
    def read_audio(self, audio_path):
        """Read the audio, change the sample rate and randomly pick one channel"""
        sig, _ = openAudioFile(audio_path, sample_rate=self.sample_rate)
        return sig

    def split_segment(self, array):
        splitted_array = splitSignal(array, rate=self.sample_rate, seconds=self.length_segments, overlap=self.overlap, minlen=3)
        return splitted_array

    def get_processed_list(self, audio_path):

        list_segments = []

        track = self.read_audio(audio_path)        
        list_divided = self.split_segment(track)
        list_segments.append(list_divided)
        return list_segments

class AudioLoader(Dataset):
    def __init__(self, list_data, sr=44100, transform=None):
        self.data = list_data
        self.transform = transform
        self.sr=sr

    def __len__(self):
        return len(self.data)

    def process_data(self, data):

        array = data
        array = array.reshape(1, -1)
        array = torch.tensor(array)
        return array

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tensor = self.process_data(self.data[idx])
        return tensor 