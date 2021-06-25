import pickle
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pylab as plt
import os
from tqdm.auto import tqdm
import optuna

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import sqlite3
from sqlalchemy import create_engine
from sklearn.metrics import f1_score
from torch.utils.data import Dataset, DataLoader
import gensim
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack




class Param_Search():

  def __init__(self, 
               model, 
               train_loader, 
               test_loader,
               criterion,
               num_epochs,
               study_name,
               n_trials=4
               ):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.criterion = criterion
    self.num_epochs = num_epochs
    self.study_name = study_name
    self.n_trials = n_trials
    self.best_model = None
    
    """Performs the hyper parameters tuning by using a TPE (Tree-structured Parzen Estimator) 
    algorithm sampler.  
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion : loss function for training the model.
    num_epochs (int): number of epochs.
    study_name (str): name of the Optuna study object.
    n_trial (int): number of trials to perform in the Optuna study.
        Default: 4
    
    Attributes:
    ------------------
    best_model: stores the weights of the common layers of the best performing model.
    
    Returns:
    ------------------
    Prints values of the optimised hyperparameters and saves the parameters of the best model.
    """
    

  def objective(self, trial):
    """Defines the objective to be optimised (F1 test score) and saves
    each final model.
    """

    # generate the model
    model = self.model

    # generate the possible optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    # convert model data type to double
    model = model.double()

    
    # Define the training and testing phases
    for epoch in tqdm(range(1, self.num_epochs + 1), desc='Epochs'):
      train_loss = 0.0
      test_loss = 0.0
      f1_test = 0.0
    
      # set the model in training modality
      model.train()
      for data, target1, target2, target3 in tqdm(self.train_loader, desc='Training Model'):
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model(data.double())
        # calculate the batch loss as a sum of the single losses
        loss = self.criterion(output1, target1) + self.criterion(output2, target2) + self.criterion(output3, target3)
        # backward pass: compute gradient of the loss wrt model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()
        
    # set the model in testing modality
      model.eval()
      for data, target1, target2, target3 in tqdm(self.test_loader, desc='Testing Model'):  
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model(data.double())
        # calculate the batch loss as a sum of the single losses
        loss = self.criterion(output1, target1) + self.criterion(output2, target2) + self.criterion(output3, target3)
        # update test loss 
        test_loss += loss.item()
        # calculate F1 test score as weighted sum of the single F1 scores
        f1_test += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) )/3

      # calculate epoch score by dividing by the number of observations
      f1_test /= (len(self.test_loader))
    
      # pass the score of the epoch to the study to monitor the intermediate objective values
      trial.report(f1_test, epoch)

    # save the final model named with the number of the trial 
    with open("{}{}.pickle".format(self.study_name, trial.number), "wb") as fout:
      pickle.dump(model, fout)
    
    # return F1 score to the study
    return f1_test



  def run_trial(self):
    """Runs Optuna study and stores the best model in class attribute 'best_model'."""
    
    # create a new study or load a pre-existing study. use sqlite backend to store the study.
    study = optuna.create_study(study_name=self.study_name, direction="maximize", 
                                storage='sqlite:///SA_optuna_tuning.db', load_if_exists=True)
    
    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    
    # if the number of already completed trials is lower than the total number of trials passed as
    #argument, perform the remaining trials 
    if len(complete_trials)<self.n_trials:
        # set the number of trials to be performed equal to the number of missing trials
        self.n_trials -= len(complete_trials)
        study.optimize(self.objective, n_trials=self.n_trials)
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        
    # store the best model found in the class
    with open("{}{}.pickle".format(self.study_name, study.best_trial.number), "rb") as fin:
        best_model = pickle.load(fin)

    self.best_model = best_model

    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
      print("    {}: {}".format(key, value))






class Param_Search_Multimodal():

  """Performs the hyper parameters tuning by using a TPE (Tree-structured Parzen Estimator) 
    algorithm sampler.  
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): dictionary of training DataLoader objects. Keys of the
        dictionary must be 'FFNN', 'CNN', 'D2V_CNN'.
    test_loader (DataLoader): dictionary of testing DataLoader objects. Keys of the
        dictionary must be 'FFNN', 'CNN', 'D2V_CNN'.
    criterion : loss function for training the model.
    num_epochs (int): number of epochs.
    study_name (str): name of the Optuna study object.
    n_trial (int): number of trials to perform in the Optuna study.
        Default: 4
    
    Attributes:
    ------------------
    best_model: stores the weights of the common layers of the best performing model.
    
    Returns:
    ------------------
  Prints values of the optimised hyperparameters and saves the parameters of the best model.
    """
    
  def __init__(self, 
               model, 
               train_loader, 
               test_loader,
               criterion,
               num_epochs,
               n_trials,
               study_name):
    self.model = model
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.criterion = criterion
    self.num_epochs = num_epochs
    self.n_trials = n_trials
    self.study_name = study_name
    self.len_train_loader = len(train_loader['FFNN'])
    self.len_test_loader = len(test_loader['FFNN'])
    self.best_model = None

  def objective(self, trial):
    """Defines the objective to be optimised (F1 test score) and saves
    each final model.
    """
    
    # generate the model
    model = self.model

    # generate the possible optimizers
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)
    
    # convert model data type to double
    model = model.double()
    
    # Define the training and testing phases
    for epoch in tqdm(range(1, num_epochs + 1)):
      train_loss = 0.0
      test_loss = 0.0
      f1_test = 0.0
    
      # set the model in training modality
      model.train()
      for load1, load2, load3 in tqdm(zip(self.train_loader['FFNN'],
                                          self.train_loader['CNN'],
                                          self.train_loader['D2V_CNN']), 
                                      desc='Training model', total = self.len_train_loader):
        x_1, target1, target2, target3 = load1
        x_2, _, _, _ = load2
        x_3, _, _, _ = load3

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model([x_1.double(), x_2.double(), x_3.double()])
        # calculate the batch loss as a sum of the single losses
        loss = self.criterion(output1, target1) + self.criterion(output2, target2) + self.criterion(output3, target3)
        # backward pass: compute gradient of the loss wrt model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()      
        
        
      # set the model in testing modality
      model.eval()
      for load1, load2, load3 in tqdm(zip(self.test_loader['FFNN'],
                                          self.test_loader['CNN'],
                                          self.test_loader['D2V_CNN']), 
                                      desc='Testing model', total = self.len_test_loader):
        x_1, target1, target2, target3 = load1
        x_2, _,_,_ = load2
        x_3, _,_,_ = load3

        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model([x_1.double(), x_2.double(), x_3.double()])
        # calculate the batch loss as a sum of the single losses
        loss = self.criterion(output1, target1) + self.criterion(output2, target2) + self.criterion(output3, target3)
        # update test loss 
        test_loss += loss.item() 
        # calculate F1 test score as weighted sum of the single F1 scores
        f1_test += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) )/3
        
      # calculate epoch score by dividing by the number of observations
      f1_test /= self.len_test_loader
      # pass the score of the epoch to the study to monitor the intermediate objective values    
      trial.report(f1_test, epoch)

    # save the final model named with the number of the trial 
    with open("{}{}.pickle".format(self.study_name,trial.number), "wb") as fout:
      pickle.dump(model, fout)
    
    # return F1 score to the study        
    return f1_test



  def run_trial(self):
    """Runs Optuna study and stores the best model in class attribute 'best_model'."""

    # create a new study or load a pre-existing study. use sqlite backend to store the study.
    study = optuna.create_study(study_name=self.study_name, direction="maximize", 
                                storage='sqlite:///SA_optuna_tuning.db', load_if_exists=True)

    complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
    
    # if the number of already completed trials is lower than the total number of trials passed as
    #argument, perform the remaining trials 
    if len(complete_trials)<self.n_trials:
        # set the number of trials to be performed equal to the number of missing trials
        self.n_trials -= len(complete_trials)
        study.optimize(self.objective, n_trials=self.n_trials)
        pruned_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.PRUNED]
        complete_trials = [t for t in study.trials if t.state == optuna.structs.TrialState.COMPLETE]
        
    # store the best model found in the class
    with open("{}{}.pickle".format(self.study_name, study.best_trial.number), "rb") as fin:
        best_model = pickle.load(fin)

    self.best_model = best_model
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
      print("    {}: {}".format(key, value))
                                          

    with open("{}{}.pickle".format(self.study_name, study.best_trial.number), "rb") as fin:
      best_model = pickle.load(fin)
    
    # store only best model
    self.best_model = best_model    




def save_best_model(self, path):
    """Saves the weights of the common layers of the best performing model.
    
    Parameters:
    ------------------
    path: path where the model will be stored.
    
    Returns:
    ------------------
    Weights of the common layers of the best model.
    """
    
    # retrieve the weights of the best model
    model_param = self.best_model.state_dict()
    
    # save only the weights of the common layers
    for key,value in model_param.copy().items():
      if re.findall('single', key):
        del model_param[str(key)]

    basepath = 'models'
    path = os.path.join(basepath, path)

    torch.save(model_param, path)

    return model_param    






class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience.
       Modified from https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    
    Parameters:
    ------------------
        patience (int): How long to wait after last time validation loss improved.
            Default: 7
        verbose (bool): If True, prints a message for each validation loss improvement. 
            Default: False
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        trace_func (function): trace print function.
            Default: print 
                            
    Attributes:
    ------------------
        early_stop (bool): True if the validation loss doesn't improveand the training should
            be stopped, False else.
        """
    
    def __init__(self, patience=3, verbose=False, delta=0, trace_func=print):
       
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss, model):

        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        # if the new score is worse than the previous score, add 1 to the counter
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            # if the number of non-improving epochs is greater than patience, 
            #set to True early_stop attribute 
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0





def fit(model, 
        train_loader, 
        test_loader, 
        criterion, 
        optimizer, 
        num_epochs, 
        filename_path, 
        patience,
        delta=0,
        verbose=True): 
    
  """Performs the training of the multitask model. It implements also early stopping
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): training DataLoader object.
    test_loader (DataLoader): testing DataLoader object.
    criterion: loss function for training the model.
    optimizer (torch.optim): optimization algorithm for training the model. 
    num_epochs (int): number of epochs.
    filename_path (str): where the weights of the model at each epoch will be stored. 
        Indicate only the name of the folder.
    patience (int): number of epochs in which the test error is not anymore decreasing
        before stopping the training.
    delta (int): minimum decrease in the test error to continue with the training.
        Default:0
    verbose (bool): prints the training error, test error, F1 training score, F1 test score 
        at each epoch.
        Default: True
    
    Attributes:
    ------------------
    f1_train_scores: stores the F1 training scores for each epoch.
    f1_test_scores: stores the F1 test scores for each epoch.
    
    Returns:
    ------------------
    Lists of F1 training scores and F1 test scores at each epoch.
    Prints training error, test error, F1 training score, F1 test score at each epoch.
    """

  basepath = 'exp'

  # keep track of epoch losses 
  f1_train_scores = []
  f1_test_scores = []

  # convert model data type to double
  model = model.double()

  # define early stopping
  early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)
    
    
  for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
    train_loss = 0.0
    test_loss = 0.0
    
    f1_train = 0.0
    f1_test = 0.0
    
    # if there is already a trained model stored for a specific epoch, load the model
    #and don't retrain the model
    PATH = os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt')
    if os.path.exists(PATH):
      checkpoint = torch.load(PATH)
      model.load_state_dict(checkpoint['model_state_dict'])
      f1_train = checkpoint['F1_train']
      f1_test = checkpoint['F1_test']
      train_loss = checkpoint['train_loss']
      test_loss = checkpoint['test_loss']
        
    else:
      # set the model in training modality
      model.train()
      for data, target1, target2, target3 in tqdm(train_loader, desc='Training model'):
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model(data.double())
        # calculate the batch loss as the sum of all the losses
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3)
        # backward pass: compute gradient of the loss wrt model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()
        # calculate F1 training score as a weighted sum of the single F1 scores
        f1_train += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) )/3
        return output1, output2, output3, target1, target2, target3

        
      # set the model in testing modality
      model.eval()
      for data, target1, target2, target3 in tqdm(test_loader, desc='Testing model'):
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model(data.double())
        # calculate the batch loss as the sum of all the losses
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3)
        # update test loss
        test_loss += loss.item()
        # calculate F1 test score as a weighted sum of the single F1 scores
        f1_test += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) ) /3
    
    
    # save the model weights, epoch, scores and losses at each epoch
    model_param = model.state_dict()
    PATH = os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt')
    torch.save({'epoch': epoch,
                'model_state_dict': model_param,
                'F1_train': f1_train,
                'F1_test': f1_test,
                'train_loss': train_loss,
                'test_loss': test_loss},
               PATH)
    
    # calculate epoch score by dividing by the number of observations
    f1_train /= (len(train_loader))
    f1_test /= (len(test_loader))
    # store epoch score
    f1_train_scores.append(f1_train)    
    f1_test_scores.append(f1_test)
      
    # print training/test statistics 
    if verbose == True:
      print('Epoch: {} \tTraining F1 score: {:.4f} \tTest F1 score: {:.4f} \tTraining Loss: {:.4f} \tTest Loss: {:.4f}'.format(
      epoch, f1_train, f1_test, train_loss, test_loss))
    
    # early stop the model if the test loss is not improving
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
      print('Early stopping the training')
      # reload the previous best model before the test loss started decreasing
      best_checkpoint = torch.load(os.path.join(basepath,filename_path + '_' + '{}'.format(epoch-patience) + '.pt'))
      model.load_state_dict(best_checkpoint['model_state_dict'])
      break
            
  
  # return the scores at each epoch
  return f1_train_scores, f1_test_scores



def fit_multimodal(model, 
                   train_loader, 
                   test_loader, 
                   criterion, 
                   optimizer, 
                   num_epochs, 
                   filename_path,
                   patience=3, 
                   delta=0,
                   verbose=True): 
  """Performs the training of the multitask model. It implements also early stopping
    
    Parameters:
    ------------------
    model (torch.nn.Module): neural network model.
    train_loader (DataLoader): dictioary of training DataLoader objects. Keys of the
        dictionary must be 'FFNN', 'CNN', 'D2V_CNN'.
    test_loader (DataLoader): dictionary of testing DataLoader objects. Keys of the
        dictionary must be 'FFNN', 'CNN', 'D2V_CNN'.
    criterion: loss function for training the model.
    optimizer (torch.optim): optimization algorithm for training the model. 
    num_epochs (int): number of epochs.
    filename_path (str): where the weights of the model at each epoch will be stored. 
        Indicate only the name of the folder.
    patience (int): number of epochs in which the test error is not anymore decreasing
        before stopping the training.
    delta (int): minimum decrease in the test error to continue with the training.
        Default:0
    verbose (bool): prints the training error, test error, F1 training score, F1 test score 
        at each epoch.
        Default: True
    
    Attributes:
    ------------------
    f1_train_scores: stores the F1 training scores for each epoch.
    f1_test_scores: stores the F1 test scores for each epoch.
    
    Returns:
    ------------------
    Lists of F1 training scores and F1 test scores at each epoch.
    Prints training error, test error, F1 training score, F1 test score at each epoch.
    """

  basepath = 'exp'

  # keep track of epoch losses 
  f1_train_scores = []
  f1_test_scores = []

  # convert model data type to double
  model = model.double()

  # define early stopping
  early_stopping = EarlyStopping(patience=patience, delta=delta, verbose=True)

  len_train_loader = len(train_loader['FFNN'])
  len_test_loader = len(test_loader['FFNN'])
    

  for epoch in tqdm(range(1, num_epochs + 1), desc='Epochs'):
    train_loss = 0.0
    test_loss = 0.0
    
    f1_train = 0.0
    f1_test = 0.0
    
    # if there is already a trained model stored for a specific epoch, load the model
    #and don't retrain the model
    PATH = os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt')
    if os.path.exists(PATH):
      checkpoint = torch.load(PATH)
      model.load_state_dict(checkpoint['model_state_dict'])
      f1_train = checkpoint['F1_train']
      f1_test = checkpoint['F1_test']
      train_loss = checkpoint['train_loss']
      test_loss = checkpoint['test_loss']
        
    else:
      # set the model in training modality
      model.train()
      for load1, load2, load3 in tqdm(zip(train_loader['FFNN'],
                                          train_loader['CNN'],
                                          train_loader['D2V_CNN']), 
                                      desc='Training model', total = len_train_loader):
        x_1, target1, target2, target3 = load1
        x_2, _,_,_ = load2
        x_3, _,_,_ = load3
        
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model([x_1.double(), x_2.double(), x_3.double()])
        # calculate the batch loss as the sum of all the losses
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3)
        # backward pass: compute gradient of the loss wrt model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss
        train_loss += loss.item()
        # calculate F1 training score as a weighted sum of the single F1 scores
        f1_train += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) )/3
        
        
      # set the model in testing modality
      model.eval()
      for load1, load2, load3 in tqdm(zip(test_loader['FFNN'],
                                          test_loader['CNN'],
                                          test_loader['D2V_CNN']), 
                                      desc='Testing model', total = len_test_loader):
        x_1, target1, target2, target3 = load1
        x_2, _,_,_ = load2
        x_3, _,_,_ = load3
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2, output3 = model([x_1.double(), x_2.double(), x_3.double()])
        # calculate the batch loss as the sum of all the losses
        loss = criterion(output1, target1) + criterion(output2, target2) + criterion(output3, target3)
        # update test loss
        test_loss += loss.item()
        # calculate F1 test score as a weighted sum of the single F1 scores
        f1_test += ( F1(output1,target1) + F1(output2,target2) + F1(output3,target3) ) /3
        
        
    # save the model weights, epoch, scores and losses at each epoch
    model_param = model.state_dict()
    PATH = os.path.join(basepath, filename_path + '_' + str(epoch) + '.pt')
    torch.save({'epoch': epoch,
                'model_state_dict': model_param,
                'F1_train': f1_train,
                'F1_test': f1_test,
                'train_loss': train_loss,
                'test_loss': test_loss},
               PATH)
     
    
    # calculate epoch score by dividing by the number of observations
    f1_train /= len_train_loader
    f1_test /= len_test_loader
    # store epoch scores
    f1_train_scores.append(f1_train)    
    f1_test_scores.append(f1_test)
      
    # print training/test statistics 
    if verbose == True:
      print('Epoch: {} \tTraining F1 score: {:.4f} \tTest F1 score: {:.4f} \tTraining Loss: {:.4f} \tTest Loss: {:.4f}'.format(
      epoch, f1_train, f1_test, train_loss, test_loss))
      
    # early stop the model if the test loss is not improving
    early_stopping(test_loss, model)
    if early_stopping.early_stop:
      print('Early stopping the training')
      # reload the previous best model before the test loss started decreasing
      best_checkpoint = torch.load(os.path.join(basepath,filename_path + '_' + '{}'.format(epoch-patience) + '.pt'))
      model.load_state_dict(best_checkpoint['model_state_dict'])
      break
    
        
  # return the scores at each epoch
  return f1_train_scores, f1_test_scores





def load_model(model, path):
  """Load the stored weights of a pre-trained model into another
      model and set it to eval state.
    
    Parameters:
    ------------------
    model (torch.nn.Module): not trained neural network model.
    path (str): path of the stored weights of the pre-trained model. 
    """

  basepath = 'models'
  path = os.path.join(basepath, path)
  checkpoint = torch.load(path)
  model.load_state_dict(checkpoint) 
  # set the model in testing modality
  model.eval() 





def save_best_model(model, path):
    """Saves only the weights of the common layers of a
    trained neural network. 
    
    Parameters:
    ------------------
    model (torch.nn.Module): trained neural network model.
    path (str): path where the weights of the trained model will be stored. 
    """
    
    model_param = model.state_dict()
    for key,value in model_param.copy().items():
      if re.findall('single', key):
        del model_param[str(key)]

    basepath = 'models'
    PATH = os.path.join(basepath, path)
    
    torch.save(model_param, PATH)





def plot_model_scores(y_train, y_test, epochs, set_ylim=None):
    """Plots the trend of the training and test loss function of 
        a model.
    
    Parameters:
    ------------------
    y_train (list): list of training losses.
    y_test (list): list of test losses.
    epochs (int): number of epochs.
    set_ylim (tuple of int): range of y-axis.
        Default: None
    """
   
    epochs = range(epochs)
    X=pd.DataFrame({'epochs':epochs,'y_train':y_train,'y_test':y_test})
   
    sns.set_theme(style="darkgrid")
    sns.set(rc={'figure.figsize':(30,15)})

    f, ax = plt.subplots(1, 1)

    sns.lineplot(data=X, x="epochs", y="y_test", color='red',lw=2.5)
    sns.lineplot(data=X, x="epochs", y="y_train", color='green',lw=2.5)

    plt.legend(labels=['F1 test score', 'F1 train score'])
    plt.setp(ax.get_legend().get_texts(), fontsize=35)
    plt.setp(ax.get_legend().get_title(),fontsize=35)

    ax.set_ylabel('F1 score', fontsize=30)
    ax.set_xlabel('Epochs', fontsize=30)
    ax.tick_params(axis="y", labelsize=20)
    ax.tick_params(axis="x", labelsize=20)
    ax.set_ylim(set_ylim)




def F1(output, target):
  pred = torch.argmax(output, dim=1)
  return f1_score(pred.cpu().detach().numpy(), target.cpu().detach().numpy(), average='weighted')


