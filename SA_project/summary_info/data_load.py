import pandas as pd
import numpy as np


class Data_Load():
    
  """
    Load the New York Times Dataset preprocessed data to be used.
        
    Returns:
    ------------------
    Dataframe of variables, Dataframe of comments.  
  """
    
  def  __init__(self):
    self.data = []
    self.original_commentBody = []
        

  def load(self):
    
    # load the data 
    self.data  = pd.read_csv('data/dataset_cleansed_version.csv', index_col='Unnamed: 0')
    
    self.original_commentBody = pd.read_csv('data/dataset_commentBody.csv', index_col='Unnamed: 0')
    
    # fix columns type
    self.data  = self.data .astype({
        'approveDate': 'float64',
        'createDate': 'float64',
        'depth': 'object',
        'picURL': 'object',
        'sharing': 'object',
        'timespeople': 'object',
        'trusted':'object',
        'updateDate': 'float64',
        'articleWordCount_x' : 'float64',
        'printPage_x' : 'object'
                    })
    
    return self.data, self.original_commentBody