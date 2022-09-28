import glob
import torch
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset

from training.lightning_trainer.sampler import ImbalancedDatasetSampler

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

        if self.transform:
            array = self.transform(array, self.sr)

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

class AudioDataModule(pl.LightningDataModule):

    def __init__(self, train_list, val_list, label_encoder, batch_size, num_workers, pin_memory, sampler=True, transform=None):
        self.batch_size = batch_size
        self.train_list = train_list
        self.val_list = val_list
        self.label_encoder = label_encoder
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.transform = transform
        self.sampler = sampler

    def train_val_loader(self):
        train_loader = AudioLoader(self.train_list, self.label_encoder, self.transform)

        if self.sampler:
            trainLoader =  DataLoader(train_loader, batch_size=self.batch_size, num_workers=self.num_workers, 
                                                pin_memory=self.pin_memory, sampler=ImbalancedDatasetSampler(train_loader))
        else:
            trainLoader =  DataLoader(train_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

        val_loader = AudioLoader(self.val_list, self.label_encoder, self.transform)
        valLoader = DataLoader(val_loader, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=self.pin_memory)

        return (trainLoader, valLoader)

