import torch
import numpy as np
import glob

from torch.utils.data import DataLoader

from model.custom_model import CustomAudioCLIP
from prediction_scripts._utils import AudioList


sample_data = "/Data/audioCLIP/tiny_training/Norwegian_soundscape/19700208_085144.sel.05.wav"

path_model = "/app/assets/ckpt-epoch=34-val_loss=0.19-lr=0.001.ckpt"
model = CustomAudioCLIP(num_target_classes=2).load_from_checkpoint(path_model, num_target_classes=2).eval()

list_preds = AudioList().get_processed_list([sample_data])
predLoader = DataLoader(list_preds, batch_size=1, num_workers=4, pin_memory=False)

proba_list = []

for array in predLoader:
    tensor = torch.tensor(array)
    output = model(tensor)
    output = np.exp(output.detach().numpy())
    print(output)
    proba_list.append(output[0])