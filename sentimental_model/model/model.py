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

# Create a function that returns a model
def create_model(activation, optimizer, loss, accuracy_metric,vocab_size,EMBEDDING_DIM,maxlen):
    model = Sequential()
    EMBEDDING_DIM = 32
    model.add(Embedding(input_dim = vocab_size, output_dim = EMBEDDING_DIM, input_length=maxlen, mask_zero=True))
    model.add(LSTM(units=40,  dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1, activation=activation))

    # Try using different optimizers and different optimizer configs
    #model.compile(loss=loss, optimizer=optimizer, metrics=[accuracy_metric])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print('Summary of the built model...')
    print(model.summary())
    return model    



