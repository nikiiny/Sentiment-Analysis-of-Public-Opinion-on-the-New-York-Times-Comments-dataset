import pandas as pd
import numpy as np
import os
import torch
import torch.nn.functional as F
import torch.nn as nn



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
  def __init__(self, concat_layer_size):
    super(MultimodalMultitask_NN, self).__init__()
    
    # input parameters
    self.concat_layer_size = concat_layer_size
    
    # VGG convolutional neural network
    self.FFNN_multitask = FFNN_multitask_pre()
    self.FFNN_multitask = nn.DataParallel(self.FFNN_multitask)

    self.CNN_multitask = CNN_multitask_pre()
    self.CNN_multitask = nn.DataParallel(self.CNN_multitask)

    self.D2V_CNN_multitask = CNN_2_multitask_pre()
    self.D2V_CNN_multitask = nn.DataParallel(self.D2V_CNN_multitask)
      

    # post concat layers

    self.post_layer1 = nn.Sequential(
        nn.Linear(self.concat_layer_size, 4000), 
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
    







