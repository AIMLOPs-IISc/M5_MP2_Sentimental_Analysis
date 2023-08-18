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

from sentimental_model.model.model import create_model 


def trian_model(model, X_train_pad, y_train, X_val_pad, y_val,batch_size,epochs,verbose):
    
    print("trian_model : start")        
    history = model.fit(X_train_pad, y_train, batch_size=batch_size, epochs=epochs, verbose=verbose, validation_data=(X_val_pad, y_val))
    
    print("history:\n",history)
    
    return model

def evaluate_model(model,X_test_pad,y_test ,batch_size):
    print('Testing...')
    y_test = np.array(y_test)
    score, acc = model.evaluate(X_test_pad, y_test, batch_size=batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    print("Accuracy: {0:.2%}".format(acc))
    return acc