import numpy as np

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer

import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from collections import Counter

from model import AudioCLIP



# See https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/domain_templates/computer_vision_fine_tuning.py
class MilestonesFinetuning(BaseFinetuning):
    def __init__(self, milestones: tuple = (5, 10), train_bn: bool = False):
        super().__init__()
        self.milestones = milestones
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module: pl.LightningModule):
        self.freeze(modules=pl_module.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer: Optimizer, opt_idx: int):
        if epoch == self.milestones[0]:
            # unfreeze 5 last layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[-5:], optimizer=optimizer, train_bn=self.train_bn
            )

        elif epoch == self.milestones[1]:
            # unfreeze remaining layers
            self.unfreeze_and_add_param_group(
                modules=pl_module.feature_extractor[:-5], optimizer=optimizer, train_bn=self.train_bn
            )

class TransferTrainingModule(pl.LightningModule):

    def __init__(self, learning_rate, num_target_classes, model_arguments=None):
        super().__init__()
        model_arguments = {} if model_arguments is None else model_arguments
        self.aclp = AudioCLIP(pretrained=f'/app/assets/AudioCLIP-Full-Training.pt')
        self.num_target_classes = num_target_classes
        print("Init model")
        self._build_model()
        self.lr = learning_rate
        self.counter = Counter()

    def _build_model(self):
        backbone = self.aclp.audio
        num_filters = backbone.fc.out_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Classifier
        self.classifier = nn.Linear(num_filters, self.num_target_classes)

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            audio_features = self.aclp.encode_audio(x) # TAKE A LOOK AT L 151 OF AUDIOCLIP
            #representations = self.feature_extractor(x)
        x = self.classifier(audio_features)
        x = F.log_softmax(x)
        return x
        
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {len(trainable_parameters)} "
            f"trainable parameters out of {len(parameters)}."
        )
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        #scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer]#, [scheduler]

    def accuracy(self, logits, labels):
        _, predicted = torch.max(logits.data, 1)
        correct = (predicted == labels).sum().item()
        accuracy = correct / len(labels)
        return torch.tensor(accuracy)

    def training_step(self, train_batch, batch_idx):

        x,y = train_batch
        self.counter.update(y.to("cpu").numpy())
        logits = self.forward(x)
        #print("Label is: {}, predicted output is: {}".format(y[0].detach().numpy(), np.exp(logits[0].detach().numpy())))
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)

        self.log('train_loss', loss)
        self.log('train_accuracy', accuracy)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        logits = self.forward(x)
        loss = self.cross_entropy_loss(logits, y)
        accuracy = self.accuracy(logits, y)
        return {'val_loss': loss, 'val_accuracy': accuracy}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        avg_acc = torch.stack([x["val_accuracy"] for x in outputs]).mean()
        self.log("val_loss", avg_loss)
        self.log("val_accuracy", avg_acc)

        print("At training, Category 0: {} samples, Category 1: {} samples".format(self.counter[0], self.counter[1]))
