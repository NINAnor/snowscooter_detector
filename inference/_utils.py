import torch
import glob
import numpy as np
import librosa
import itertools

from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

RANDOM = np.random.RandomState(42)

def openAudioFile(path, sample_rate=44100, offset=0.0, duration=None):    

    try:
        sig, rate = librosa.load(path, sr=sample_rate, offset=offset, duration=duration, mono=True, res_type='kaiser_fast')
    except:
        sig, rate = [], sample_rate

    return sig, rate

def splitSignal(sig, rate, seconds, overlap, minlen):

    # Split signal with overlap
    sig_splits = []
    for i in range(0, len(sig), int((seconds - overlap) * rate)):
        split = sig[i:i + int(seconds * rate)]

        # End of signal?
        if len(split) < int(minlen * rate):
            break
        
        # Signal chunk too short?
        if len(split) < int(rate * seconds):
            split = np.hstack((split, noise(split, (int(rate * seconds) - len(split)), 0.5)))
        
        sig_splits.append(split)

    return sig_splits

def noise(sig, shape, amount=None):

    # Random noise intensity
    if amount == None:
        amount = RANDOM.uniform(0.1, 0.5)

    # Create Gaussian noise
    try:
        noise = RANDOM.normal(min(sig) * amount, max(sig) * amount, shape)
    except:
        noise = np.zeros(shape)

    return noise.astype('float32')

class AudioList():

    def __init__(self, length_segments = 3, sample_rate=44100, overlap=0):
        self.sample_rate = sample_rate
        self.length_segments = length_segments
        self.overlap = overlap

    def normalize_audio(self, list_segments):

        normalized_array=[]
        for array in list_segments:
            normalized_array.append(array)

        return normalized_array

    def read_audio(self, audio_path):
        """Read the audio, change the sample rate and randomly pick one channel"""
        sig, _ = openAudioFile(audio_path, sample_rate=self.sample_rate)
        return sig

    def split_segment(self, array):
        splitted_array = splitSignal(array, rate=self.sample_rate, 
                                    seconds=self.length_segments, 
                                    overlap=self.overlap, 
                                    minlen=self.length_segments)
        return splitted_array

    def get_processed_list(self, audio_path):

        track = self.read_audio(audio_path)        
        list_divided = self.split_segment(track)
        #list_segments_norm = self.normalize_audio(list_divided)
        return list_divided

class AudioListMetrics():

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

class AudioLoader(Dataset):
    def __init__(self, list_data, label_encoder, sr=44100, transform=None):
        self.data = list_data
        self.label_encoder = label_encoder
        self.transform = transform
        self.sr=sr

    def __len__(self):
        return len(self.data)

    def process_data(self, data):

        array, label = data
        array = array.reshape(1, -1)
        array = torch.tensor(array)

        label_encoded = self.label_encoder.one_hot_sample(label)
        label_class = torch.argmax(label_encoded)

        return (array, label_class)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tensor, label = self.process_data(self.data[idx])
        return tensor, label

    def get_labels(self):
        list_labels = []
        for x,y in self.data:
            list_labels.append(y)
        return list_labels

class EncodeLabels():
    """
    Function that encodes names of folders as numerical labels
    Wrapper around sklearn's LabelEncoder
    """
    def __init__(self, path_to_folders):
        self.path_to_folders = path_to_folders
        self.class_encode = LabelEncoder()
        self._labels_name()

    def _labels_name(self):
        labels = glob.glob(self.path_to_folders + "/*")
        labels = [l.split("/")[-1] for l in labels]
        self.class_encode.fit(labels)
        
    def __getLabels__(self):
        return self.class_encode.classes_

    def to_one_hot(self, codec, values):
        value_idxs = codec.transform(values)
        return torch.eye(len(codec.classes_))[value_idxs]

    def one_hot_sample(self, label):
        t_label = self.to_one_hot(self.class_encode, [label])
        return t_label

