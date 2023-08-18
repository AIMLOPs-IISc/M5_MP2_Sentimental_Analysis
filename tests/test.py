"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
from sentimental_model import __version__ as _version
from sentimental_model.config.core import config
#from sentiment_model.predict import make_prediction
from sentimental_model.processing.data_manager import load_dataset , split_data, convert_text_to_int_values

from sentimental_model.model.model import create_model 
from sentimental_model.model.train_model import evaluate_model, trian_model

import numpy as np
import tensorflow as tf
from sentimental_model import __version__ as _version
from sentimental_model.config.core import config
#from sentiment_model.predict import make_prediction
from sentimental_model.processing.data_manager import load_dataset , split_data, convert_text_to_int_values

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
import io
import re
import json
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers

def test(sample_input_data):
   
    # read training data
    data = sample_input_data
    
    print("data:size:",len(data))
    print(data.head(10))
    
    X_train, X_val, X_test,y_train, y_val, y_test = split_data(
        data, 
        config.model_config.feature_name,
        config.model_config.label_name,
        config.model_config.split_1,
        config.model_config.split_2,
        config.model_config.random_state )
    
    X_train_pad, X_val_pad, X_test_pad,vocab_size = convert_text_to_int_values(
        X_train,
        X_val,
        X_test,
        config.model_config.num_words,
        config.model_config.maxlen,
        config.model_config.padding_val,
        config.model_config.padding_val
        )
    
    print("X_train_pad, X_val_pad, X_test_pad,vocab_size:",len(X_train_pad), len(X_val_pad), len(X_test_pad),vocab_size)

    model = create_model(config.model_config.activation, 
                          config.model_config.optimizer, 
                          config.model_config.loss, 
                          [config.model_config.accuracy_metric],
                          vocab_size,
                          config.model_config.EMBEDDING_DIM,
                          config.model_config.maxlen
                          )
         
    model_1 = trian_model(model, 
                X_train_pad, 
                y_train, 
                X_val_pad, 
                y_val,
                config.model_config.batch_size,
                config.model_config.epochs,
                config.model_config.verbose)
    
    acc = evaluate_model(model_1,X_test_pad,y_test ,config.model_config.batch_size)
    
    assert 1
