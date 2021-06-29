#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import random
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def plot_loss(model, losses, label):
    plt.figure(figsize=(10,5))
    plt.title("Training {}".format(label))
    plt.plot(losses,label=label)
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("{}-train-losses.png".format(model))
    plt.show()
    
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed) 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def save_objs(model, epoch, loss, optimizer, save_path):
    path = save_path + 'model.{}'.format(epoch)   
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        "loss": loss,
    }, path)
    
def initialize_weights(model):
  if isinstance(model, nn.Conv2d):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data,nonlinearity='relu')
      if model.bias is not None:
          nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.BatchNorm2d):
      if model.weight is not None:
        nn.init.constant_(model.weight.data, 1)
      if model.bias is not None:  
        nn.init.constant_(model.bias.data, 0)
  elif isinstance(model, nn.Linear):
      if model.weight is not None:
        nn.init.kaiming_uniform_(model.weight.data)
      if model.bias is not None:
        nn.init.constant_(model.bias.data, 0)

def plot_confusion_matirx(y_true, y_pred, labels):
    
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true')
    
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['rest','focus'])

    
    # NOTE: Fill all variables here with default values of the plot_confusion_matrix
    disp = disp.plot(include_values=True,
                     cmap=plt.cm.Blues, ax=None, xticks_rotation='horizontal')
    
    plt.show()
    