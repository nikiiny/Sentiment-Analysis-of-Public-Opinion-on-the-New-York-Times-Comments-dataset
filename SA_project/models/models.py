import pickle
import re
import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class FFNN_multitask(nn.Module):
  """Multi-task Feed Forward neural network.
      2 shared layers and a single hidden layer for each of the
      3 tasks. It uses ReLU activation functions."""

  def __init__(self):
    super(FFNN_multitask, self).__init__()
    
    self.layer1 = nn.Sequential(
        nn.Linear(261, 4000), 
        nn.ReLU(),
        nn.Linear(4000, 2000),
        nn.ReLU()) 

    self.single_layer2_1 = nn.Sequential(
    nn.Linear(2000, 1000),
    nn.ReLU())

    self.single_layer2_2 = nn.Sequential(
    nn.Linear(2000, 1000),
    nn.ReLU())

    self.single_layer2_3 = nn.Sequential(
    nn.Linear(2000, 1000),
    nn.ReLU())

    self.single_last_layer1 = nn.Linear(1000, 2) 
    self.single_last_layer2 = nn.Linear(1000, 2)
    self.single_last_layer3 = nn.Linear(1000, 2)  

    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4) 

  def forward(self, x):
      
      # 2 shared blocks 
    out = self.layer1(x)
    out = self.drop_out1(out)

      # single hidden layers for each task
    out1 = self.single_layer2_1(out)
    out1 = self.drop_out2(out1)
    out1 = self.single_last_layer1(out1)

    out2 = self.single_layer2_2(out) 
    out2 = self.drop_out2(out2)
    out2 = self.single_last_layer2(out2)

    out3 = self.single_layer2_3(out) 
    out3 = self.drop_out2(out3)
    out3 = self.single_last_layer3(out3)
 
    return out1, out2, out3




class CNN_multitask(nn.Module):
  """Multi-task Convolutional neural network.
      2 shared convolutions and a single convolution followed by a fully connected
      layer for each of the 3 tasks. It uses ReLU activation functions.
      
      Parameters
      --------------
      fc_layer_size (int): size of the fully connected layer
      """

  def __init__(self, fc_layer_size):
    super(CNN_multitask, self).__init__()
    self.fc_layer_size = fc_layer_size 
    
    self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=1), #The average word length in English language is 4.7 characters.
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))

    self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
        
    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4)
    self.drop_out3 = nn.Dropout(p=0.5)
    
    self.single_last_layer1_1 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_1 = nn.Linear(1000, 5)

    self.single_last_layer1_2 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_2 = nn.Linear(1000, 5)

    self.single_last_layer1_3 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_3 = nn.Linear(1000, 5)


  def forward(self, x):
      
      # 2 shared blocks 
    out = self.layer1(x)
    out = self.drop_out1(out)
    out = self.layer2(out)
    out = self.drop_out2(out)

      # single convolution + fully connected layer for each task
    out1 = self.single_layer3_1(out)
    out1 = self.drop_out3(out1)
    out1 = out1.reshape(out1.size(0), -1) 
    out1 = self.single_last_layer1_1(out1)
    out1 = self.single_last_layer2_1(out1)

    out2 = self.single_layer3_2(out) 
    out2 = self.drop_out2(out2)
    out2 = out2.reshape(out2.size(0), -1) 
    out2 = self.single_last_layer1_2(out2)
    out2 = self.single_last_layer2_2(out2)

    out3 = self.single_layer3_3(out) 
    out3 = self.drop_out3(out3)
    out3 = out3.reshape(out3.size(0), -1) 
    out3 = self.single_last_layer1_3(out3)
    out3 = self.single_last_layer2_3(out3)

 
    return out1, out2, out3




import torch.nn as nn

class CNN_2_multitask(nn.Module):
  """Multi-task Convolutional neural network.
      2 shared convolutions and a single convolution followed by a fully connected
      layer for each of the 3 tasks. It uses ReLU activation functions.
      
      Parameters
      --------------
      fc_layer_size (int): size of the fully connected layer
  """
  def __init__(self, fc_layer_size):
    super(CNN_2_multitask, self).__init__()
    self.fc_layer_size = fc_layer_size 
    
    self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=1), #The average word length in English language is 4.7 characters.
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2))

    self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_1 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
    
    self.single_layer3_3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
        
    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4)
    self.drop_out3 = nn.Dropout(p=0.5)
    
    self.single_last_layer1_1 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_1 = nn.Linear(1000, 5)

    self.single_last_layer1_2 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_2 = nn.Linear(1000, 5)

    self.single_last_layer1_3 = nn.Linear(self.fc_layer_size, 1000) 
    self.single_last_layer2_3 = nn.Linear(1000, 5)


  def forward(self, x):
      
      # 2 shared blocks
    out = self.layer1(x)
    out = self.drop_out1(out)
    out = self.layer2(out)
    out = self.drop_out2(out)

      # single convolution + fully connected layer for each task
    out1 = self.single_layer3_1(out)
    out1 = self.drop_out3(out1)
    out1 = out1.reshape(out1.size(0), -1) 
    out1 = self.single_last_layer1_1(out1)
    out1 = self.single_last_layer2_1(out1)

    out2 = self.single_layer3_2(out) 
    out2 = self.drop_out2(out2)
    out2 = out2.reshape(out2.size(0), -1) 
    out2 = self.single_last_layer1_2(out2)
    out2 = self.single_last_layer2_2(out2)

    out3 = self.single_layer3_3(out) 
    out3 = self.drop_out3(out3)
    out3 = out3.reshape(out3.size(0), -1) 
    out3 = self.single_last_layer1_3(out3)
    out3 = self.single_last_layer2_3(out3)

 
    return out1, out2, out3





class FFNN_multitask_pre(nn.Module):
  def __init__(self):
    super(FFNN_multitask_pre, self).__init__()
    
    self.layer1 = nn.Sequential(
        nn.Linear(261, 4000), 
        nn.ReLU(),
        nn.Linear(4000, 2000),
        nn.ReLU()) 

    self.drop_out1 = nn.Dropout(p=0.3)

  def forward(self, x):
      
      # first block in common
    out = self.layer1(x)
    out = self.drop_out1(out)
    out = out.reshape(out.size(0), -1) 
 
    return out





class CNN_multitask_pre(nn.Module):
  def __init__(self):
    super(CNN_multitask_pre, self).__init__()
    
    self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=1), #The average word length in English language is 4.7 characters.
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))

    self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=15, stride=2, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=2))
        
    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4)


  def forward(self, x):
      
      # first blocks in common
    out = self.layer1(x)
    out = self.drop_out1(out)
    out = self.layer2(out)
    out = self.drop_out2(out)
    out = out.reshape(out.size(0), -1) 
 
    return out




class CNN_2_multitask_pre(nn.Module):
  def __init__(self):
    super(CNN_2_multitask_pre, self).__init__()
    
    self.layer1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=1), #The average word length in English language is 4.7 characters.
            nn.BatchNorm1d(32), 
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2))

    self.layer2 = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=5, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2))
        
    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4)


  def forward(self, x):
      
      # first blocks in common
    out = self.layer1(x)
    out = self.drop_out1(out)
    out = self.layer2(out)
    out = self.drop_out2(out)
    out = out.reshape(out.size(0), -1) 
 
    return out




class MultimodalMultitask_NN(nn.Module):
  def __init__(self, concat_layer_size, hyperparameters_tuning=False):
    super(MultimodalMultitask_NN, self).__init__()
    #Like in other object-oriented languages, it allows you to call 
    #methods of the superclass in your subclass. The primary use case of 
    #this is to extend the functionality of the inherited method.
    
    # input parameters
    self.hyperparameters_tuning = hyperparameters_tuning
    self.concat_layer_size = concat_layer_size
    
    # VGG convolutional neural network
    self.FFNN_multitask = FFNN_multitask_pre()
    self.FFNN_multitask = nn.DataParallel(self.FFNN_multitask)
    self.FFNN_multitask.to(device)

    self.CNN_multitask = CNN_multitask_pre()
    self.CNN_multitask = nn.DataParallel(self.CNN_multitask)
    self.CNN_multitask.to(device)

    self.D2V_CNN_multitask = CNN_2_multitask_pre()
    self.D2V_CNN_multitask = nn.DataParallel(self.D2V_CNN_multitask)
    self.D2V_CNN_multitask.to(device)

    # load previously trained models to find optimal hyperparameters
    if self.hyperparameters_tuning:
      load_model(self.FFNN_multitask, 'FFNN/best_model_FFNN_hp.pt')
      load_model(self.CNN_multitask, 'CNN/best_model_CNN_hp.pt')
      load_model(self.D2V_CNN_multitask, 'D2V_CNN/best_model_D2V_CNN_hp.pt')
    
    # load previously trained models for final testing
    else:
      load_model(self.FFNN_multitask, 'FFNN/best_model_FFNN_test.pt')
      load_model(self.CNN_multitask, 'CNN/best_model_CNN_test.pt')
      load_model(self.D2V_CNN_multitask, 'D2V_CNN/best_model_D2V_CNN_test.pt')

    # freeze layers
    for param in self.FFNN_multitask.parameters():
      param.requires_grad = False
    for param in self.CNN_multitask.parameters():
      param.requires_grad = False
    for param in self.D2V_CNN_multitask.parameters():
      param.requires_grad = False
      

    # post concat layers

    self.post_layer1 = nn.Sequential(
        nn.Linear(self.concat_layer_size, 4000), # 21072
        nn.ReLU()) 

    self.single_post_layer2_1 = nn.Sequential(
    nn.Linear(4000, 1000),
    nn.ReLU())

    self.single_post_layer2_2 = nn.Sequential(
    nn.Linear(4000, 1000),
    nn.ReLU())

    self.single_post_layer2_3 = nn.Sequential(
    nn.Linear(4000, 1000),
    nn.ReLU())

    self.single_post_last_layer1 = nn.Linear(1000, 2) 
    self.single_post_last_layer2 = nn.Linear(1000, 2)
    self.single_post_last_layer3 = nn.Linear(1000, 2)  

    self.drop_out1 = nn.Dropout(p=0.3)
    self.drop_out2 = nn.Dropout(p=0.4) 

  
  def forward(self, x):

    x_1, x_2, x_3 = x

    out_1 = self.FFNN_multitask(x_1)
    out_2 = self.CNN_multitask(x_2)
    out_3 = self.D2V_CNN_multitask(x_3)

    # concat layer
    out = torch.cat((out_1, out_2, out_3), dim=1)

    # final layer in common
    out = self.post_layer1(out)
    out = self.drop_out1(out)

    # final single layers
    out1 = self.single_post_layer2_1(out)
    out1 = self.drop_out2(out1)
    out1 = self.single_post_last_layer1(out1)

    out2 = self.single_post_layer2_2(out)
    out2 = self.drop_out2(out2)
    out2 = self.single_post_last_layer2(out2)

    out3 = self.single_post_layer2_3(out)
    out3 = self.drop_out2(out3)
    out3 = self.single_post_last_layer3(out3)

    # output softmax
    return out1, out2, out3
    
   # return nn.functional.log_softmax(output, dim=-1) #not needed since it's already applied
   #by cross-entropy loss

