
import pandas as pd
import numpy as np

import os
import pickle
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import gensim

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler
from scipy.sparse import csr_matrix
from scipy.sparse import coo_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
import types




class Data_Load():
    
  """
    Load the New York Times Dataset preprocessed data to be used.
        
    Parameters:
    ------------------
    data_path (str): path of the dataset.
        
    Raises:
    ------------------
    ValueError,
        If network_type parameter value is wrong.
        
    Returns:
    ------------------
    Dataframe of variables, Dataframe of labels.  
  """
    
  def  __init__(self, 
               data_path):
    self.data_path = data_path
    self.X = []
    self.y =  []
        

  def load(self):
    
    # load the labels
    self.y =  pd.read_csv(
      os.path.join('data', self.data_path),
      usecols = ['editorsSelection_TARGET','recommendations_TARGET','replyCount_TARGET']
      ) 
    # fix columns type
    self.y  = self.y .astype({
        'editorsSelection_TARGET': 'int',
        'recommendations_TARGET': 'int',
        'replyCount_TARGET': 'int'})
 
    
    # load the data features based on the chosen neural network
    self.X  = pd.read_csv(
        os.path.join('data',self.data_path),
        usecols = ['approveDate','commentType','createDate','depth','picURL','sharing','timespeople',
                  'trusted','updateDate','userTitle','sectionName_x','newDesk_x','articleWordCount_x','printPage_x',
                  'typeOfMaterial_x','documentType','pubDate','source','keywords','commentBody']
        ) 
    # fix columns type
    self.X  = self.X .astype({
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
    
    return self.X, self.y





class Data_Preprocess():
  """Preprocess the New York Times Dataset data depending on the data type and the specific network used among
        feed forward, convolutional and doc2vec + convolutional neural network.
        Variables used for feed forward NN are scaled with a robust scaler (numeric) or one-hot-encoded (factors).
        The variable 'keywords' is vectorised by using TF-IDF. The variable 'commentBody' is embedded by using
        doc2vec representation.
        
    Parameters:
    ------------------
    X (pd.DataFrame): dataframe of variables.
    y (pd.DataFrame): dataframe of labels.
        
    Raises:
    ------------------
    ValueError,
        If 'network_type' parameter value is wrong.
    ValueError,
        If 'usage' parameter value is wrong.
        
    Attributes:
    ------------------
    scaler_onehot_dict (dict): it stores for each column a fitted RobustScaler() object for numeric
        variables and a fitted OneHotEncoder() object for categorical variables
    vectorizer: it stores the fitted TfidfVectorizer() object for the variable 'keywords'
    doc2vec_model: it stores the learned embedded representation of the variable 'commentBody'
        by using the doc2vec model. The window for the context is 3. The resulting vector 
        has dimensionality 300.
    
    Returns:
    ------------------
    Tensor of training data, Tensor of training labels, Tensor of test data, Tensor of test labels.
  """
    
  def __init__(self, 
               X,
               y):
  
    self.X1 = X[['approveDate','commentType','createDate','depth','picURL','sharing','timespeople',
                      'trusted','updateDate','userTitle','sectionName_x','newDesk_x','articleWordCount_x','printPage_x',
                      'typeOfMaterial_x','documentType','pubDate','source']]
    self.X2 = X['keywords']
    self.X3 = X['commentBody']
    self.y = y

    # instantiate the attributes for storing the fitted transformation methods
    self.scaler_onehot_dict = {}
    self.vectorizer = []
    self.doc2vec_model = []
      
    self.X_train = []
    self.y_train = []
    self.X_val = []
    self.y_val = []
    self.X_test = []
    self.y_test = []


  def data_process_flat_features(self):
    """Fit a robust scaler for numeric features and one-hot encoding for factors,
    then store it into a dictionary for each column. 
    """
    
    for col in sorted(self.X1.columns):
        # fit robust scaler method on each numeric column and store it into a vocabulary
        if self.X1[col].dtype == object:
            self.scaler_onehot_dict[col] = OneHotEncoder().fit(self.X1[col].values.astype('U').reshape(-1, 1))
        # fit one-hot encoder method on each categorical column and store it into a vocabulary
        elif self.X1[col].dtype == np.float64:
            self.scaler_onehot_dict[col] = RobustScaler().fit(self.X1[col].values.astype('U').reshape(-1,1))
    

  def data_process_tfidf_doc_embedding(self):
    """Fit TF-IDF vectorization method.
    """
    
    # fit TF-IDF vectoriser and store it
    self.vectorizer = TfidfVectorizer() 
    self.vectorizer.fit(self.X2.values.astype('U'))

    
  def tag_docs(self):
    """Returns a list of TaggedDocuement containing the words and a unique tag for the document
    to be used as input for the doc2vec model.
    """
    for i, line in enumerate(self.X3):
      tokens = gensim.utils.simple_preprocess(str(line))
      yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

  def data_process_doc2vec_embedding(self):
    """Learns an embedded representation by using a doc2vec model. The window for the context is 3. 
    The resulting vector has dimensionality 300. Data are not directly transformed to save memory.
    The class attribute 'doc2vec_model' stores the resulting learned model.
    """
    
    X = list(self.tag_docs())
    
    if os.path.exists("misc/doc2vec.pickle"):
      with open("misc/doc2vec.pickle", "rb") as fin:
        model = pickle.load(fin)
    
    else:        
      # build doc2vec representation of documents with dimensionality 300 and window=3
      model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=1, epochs=5, dbow_words=1, window=3) 
      # train model by passing the tagged document and words
      model.build_vocab(X)
      model.train(X, total_examples=model.corpus_count, epochs=model.epochs)
    
      with open("misc/doc2vec.pickle", "wb") as fout:
        pickle.dump(model, fout)

    self.doc2vec_model = model
        

  def split_data(self, X, usage, test_size, validation_size):
    """It splits the dataset into training and test set, if usage is 'model_testing'. Else if usage
     is 'hyper_tuning', it splits the trianing set again into a training and validation set, discarding
     the test set.
    """
    
    assert (X.shape[0] == self.y.shape[0])
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, self.y,
                                                        test_size=test_size, 
                                                        shuffle=True) 


    if usage == 'hyper_tuning':
      assert (self.X_train.shape[0] == self.y_train.shape[0])

      self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_train, self.y_train,
                                                        test_size=validation_size, 
                                                        shuffle=True) 
        
    # reset indexes of dataframe
    self.X_train = self.X_train.reset_index(drop=True)
    self.X_test = self.X_test.reset_index(drop=True)
    self.y_train = self.y_train.reset_index(drop=True)
    self.y_test = self.y_test.reset_index(drop=True)
    
    
  def fit(self):
    """Apply trasformations depending on the data
    """

    # fit transformation methods if not aready fitted
    if not self.scaler_onehot_dict:
        self.data_process_flat_features()
    if not self.vectorizer:
        self.data_process_tfidf_doc_embedding()
    if not self.doc2vec_model:
        self.data_process_doc2vec_embedding()
            
    
  def get_data(self, network_type, usage, test_size=0.25, validation_size=0.15):
    """Split the dataset and returns data depending on the neural network used.
    
    Parameters: 
    ----------------
    network_type (str): type of neural network to be used. Possible values are:
        'FFNN' (feed forward), CNN (convolutional), D2V_CNN (doc2vec + convolutional)
        Default: None
    usage (str): either hyperparameters tuning or model testing. In the first case it creates a training and 
        validation set, leaving out the test set. In the seconc case it creates a training and test set.
        Possible values are: 'hyper_tuning' or 'model_testing'
        Default: None
    test_size (float): size of the test set.
        Default: 0.25
    validation_size (float): size of the validation set.
        Default: 0.15
        
        
    Returns: 
    ----------------
        Tensor of training data, Tensor of training labels, Tensor of test data, Tensor of test labels. 
    """
    
    if network_type not in ['FFNN','CNN','D2V_CNN']:
            raise ValueError(
            "Argument 'network_type' has an incorrect value: use 'FFNN', 'CNN', 'D2V_CNN'")

    
    if usage not in ['hyper_tuning', 'model_testing']:
            raise ValueError(
            "Argument 'usage' has an incorrect value: use 'hyper_tuning', 'model_testing'")
            
    # split into training and test set
    if network_type == 'FFNN':
        self.split_data(self.X1, usage, test_size, validation_size)
    if network_type == 'CNN':
        self.split_data(self.X2, usage, test_size, validation_size)
    if network_type == 'D2V_CNN':
        self.split_data(self.X3, usage, test_size, validation_size)

        
    return self.X_train, self.y_train, self.X_test, self.y_test





class Dataset_Wrap(Dataset):
    
  """Creates a Dataset object for building a DataLoader object.
        
    Parameters:
    ------------------
    X (torch.Tensor): tensor of variables.
    y (torch.Tensor): tensor of labels.
    network_type (str): type of neural network to be used. Possible values are:
        'FFNN' (feed forward), CNN (convolutional), D2V_CNN (word embeddings + convolutional)
    scaler_onehot_dict: fitted robust scaler for numeric variables and fitted one-hot encoder
        for categorical values.
        Default: None
    vectorizer: fitted TF-IDF model
        Default: None
    doc2vec_model: learned representation of the variable 'commentBody' through doc2vec model
        Default: None
    
    Returns:
    ----------------
        i-th data and labels. Move the tensors to GPU if available
  """
    
  def __init__(self, X, y, network_type, scaler_onehot_dict = None, vectorizer = None, doc2vec_model=None):
    self.X = X
    self.y = y
    self.network_type = network_type
    self.vectorizer = vectorizer
    self.scaler_onehot_dict =  scaler_onehot_dict
    self.doc2vec_model = doc2vec_model

  def __len__(self):
    """Returns the number of observations."""
    return (self.X.shape[0])  

  def __getitem__(self, i):
    """Returns:
    ----------------
        i-th data and labels depending on the model used. 
        Move the tensors to GPU if available, otherwise use CPU."""
    
    if self.network_type == 'FFNN':

        data = torch.empty(0)
        
        for col in sorted(self.X.columns):
            x = self.scaler_onehot_dict[col].transform(np.array(self.X[col][i]).astype('U').reshape(-1, 1))
            data = hstack((data,x))        

        data = torch.tensor(data.todense()).reshape(-1)
        
        
    if self.network_type == 'CNN':
        
        data = self.vectorizer.transform([str(self.X[i])])
        data = torch.tensor( data.todense().reshape(1,-1) )
        
    if self.network_type == 'D2V_CNN':
        
        data = torch.tensor(self.doc2vec_model.infer_vector( str(self.X[i] ).split()))
        data = torch.reshape(data, (1, len(data)))

    # return labels
    labels = torch.tensor(self.y.values[i].astype(int))
    y_1, y_2, y_3 = labels
          
    return (data.to(device), y_1.to(device), y_2.to(device), y_3.to(device))





def build_DataLoader(
    data_processed, 
    network_type, 
    usage, 
    test_size=0.25, 
    validation_size=0.15,
    batch_size = 100):
    
        """Build a function that process data and return a DataLoader object that can be directly passed as 
        input to the neural network.

        Parameters:
        ------------------
        processed_data: object of class Data_Preprocess() where the method fit has been
            already called.
        network_type (str): type of neural network to be used. Possible values are:
            'FFNN' (feed forward), CNN (convolutional), EMB_CNN (word embeddings + convolutional)
            Default: None
        usage (str): either hyperparameters tuning or model testing. In the first case it creates a training and 
            validation set, leaving out the test set. In the seconc case it creates a training and test set.
            Possible values are: 'hyper_tuning' or 'model_testing'
            Default: None
        test_size (float): size of the test set.
            Default: 0.25
        validation_size (float): size of the validation set.
            Default: 0.15
        batch_size (int): size of the training batches. The size of the testing batches is doubled.
            Default: 100

        Returns:
        ------------------
        Training DataLoader object, Testing DataLoader object
        """

        X_train, y_train, X_test, y_test = data_processed.get_data(network_type=network_type,
                                                               usage=usage, 
                                                               test_size=test_size,
                                                               validation_size=validation_size)


        # create data wrapper
        train_wrap = Dataset_Wrap(X_train, y_train, network_type=network_type, 
                                  scaler_onehot_dict=data_processed.scaler_onehot_dict,
                                  vectorizer=data_processed.vectorizer,
                                  doc2vec_model=data_processed.doc2vec_model)

        test_wrap = Dataset_Wrap(X_test, y_test, network_type=network_type, 
                                 scaler_onehot_dict=data_processed.scaler_onehot_dict,
                                 vectorizer=data_processed.vectorizer,
                                 doc2vec_model=data_processed.doc2vec_model)

        # create DataLoader object
        loader_train = DataLoader(dataset = train_wrap, batch_size = batch_size, shuffle=True)             
        loader_test = DataLoader(dataset = test_wrap, batch_size = batch_size*2, shuffle=False)     

        return  loader_train, loader_test

