import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import typing as t
from pathlib import Path
import pandas as pd
import tensorflow as tf
from tensorflow import keras
#from keras.utils import image_dataset_from_directory
from sentimental_model.config.core import config
from sentimental_model import __version__ as _version
from sentimental_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
import re


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
from nltk.tokenize import word_tokenize

def _load_raw_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    return dataframe

def load_dataset(*, file_name: str) -> pd.DataFrame:
    print ("load_dataset:DATASET_DIR=",DATASET_DIR)
    print ("load_dataset:file_name=",file_name)
    
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    
    
    transformed = preprocess_data(dataframe)

    return transformed

def preprocess_data(df_new) -> pd.DataFrame:
    # Add centimaent column
    df_new['Sentiment'] = df_new['Score'].apply(lambda x:'P' if x > 2 else 'N')
    # remove duplicates
    #df_new = df_new.drop_duplicates(subset=['Sentiment', 'Text'], keep='last')
    df_new.drop_duplicates(subset=['Sentiment','Text'],inplace=True)  
    # apply the function defined above and save the
    df_new['Text'] = df_new['Text'].apply(cleanText)
    
    # apply function on cleaned resume to remove stopwords
    df_new['Text'] = df_new['Text'].apply(remove_stopwords)
    
    print("Sentiment TYpe::", type(df_new['Sentiment'].head(1)))
    
    df_new = convert_label_encoder(df_new)
    
    print("Target Encoded::", df_new['Target'].head(1))
    
    return df_new



def cleanText(Text):
  Text = Text.lower() # converting to lower case
#   Text = re.sub(r'http\S+', '', Text,flags = re.MULTILINE) # remove URLs
#   Text = re.sub('RT|cc', '', Text)  # remove RT and cc
#   Text = re.sub('#\S+', '', Text)  # remove hashtags
#   Text = re.sub('@\S+', '', Text)  # remove mentions
#   Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), '', Text)  # remove punctuations
#   Text = re.sub('â\S+', '', Text)  # remove â¢
#   Text = re.sub('+', '', Text)  # remove 
#   Text = re.sub('\s+', ' ', Text)  # remove extra whitespace
  http_pattern = "http[s]?://\S+"
  Text = re.sub(http_pattern, '', Text)   # Removing URLs
  Text = re.sub('RT|cc', '', Text)  # remove RT and cc
  Text = re.sub('#\S+', '', Text)  # remove hashtags
  Text = re.sub(r'[^x00-x7f]',r' ', Text)
  Text = re.sub('@\S+', '', Text)  # remove mentions
  Text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', Text)  # remove punctuations  
  return Text

def get_stopwards():
    #Get topwordlist
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list = [word for word in stopword_list if "n't" not in word]
    negative = ["no", "not", "nor", "don", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren", "wouldn", "won"]
    stopword_list = [word for word in stopword_list if word not in negative]
    #print(stopword_list)
    return stopword_list

def remove_stopwords(text, is_lower_case=False):
    # splitting strings into tokens (list of words)
    stopword_list = get_stopwards()
    tokens = word_tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def split_data(df_new,feature,target,split_1,split_2,random_state ):
    # Split the dataset into training and testing sets
    X_train_temp, X_test, y_train_temp, y_test = train_test_split(df_new[feature].values, df_new[target].values,
                                                        test_size=split_1,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_temp, y_train_temp,
                                                        test_size=split_2,
                                                        random_state=random_state)
    
    tot_rec=len(df_new)

    print("num_train_samples::","\t", len(X_train), "\t:: Split %=",round(((len(X_train)/tot_rec)*100)))
    print("num_val_samples::","\t", len(X_val),"\t\t:: Split %=",round(((len(X_val)/tot_rec)*100)))
    print("num_test_samples::","\t", len(X_test),"\t\t:: Split %=",round(((len(X_test)/tot_rec)*100)))

  
    return X_train, X_val, X_test,y_train, y_val, y_test

def convert_text_to_int_values(X_train,X_val,X_test,num_words,maxlen,padding_val,truncating_val):
    print("convert_text_to_int_values::X_train,X_val,X_test,num_words,maxlen,padding_val,truncating_val",X_train,X_val,X_test,num_words,maxlen,padding_val,truncating_val)
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(X_train)

    X_train_tok = tokenizer.texts_to_sequences(X_train)
    X_val_tok = tokenizer.texts_to_sequences(X_val)
    X_test_tok = tokenizer.texts_to_sequences(X_test)
 
    # Find the vocabulary size and perform padding on both train and test set
    vocab_size = len(tokenizer.word_index) + 1

    #maxlen = 100

    X_train_pad = pad_sequences(X_train_tok, padding=padding_val, maxlen=maxlen, truncating=truncating_val)
    X_val_pad = pad_sequences(X_val_tok, padding=padding_val, maxlen=maxlen, truncating=truncating_val)
    X_test_pad = pad_sequences(X_test_tok, padding=padding_val, maxlen=maxlen, truncating=truncating_val)
    
    return X_train_pad,X_val_pad,X_test_pad,vocab_size

def convert_label_encoder(df_new):
    from sklearn.preprocessing import LabelEncoder
    #targetLabels  = ['P','N']
    targetLabels  = config.model_config.lable_target_list

    le = LabelEncoder()
    le.fit(targetLabels)
    print("le.classes_:",le.classes_)
    df_new[config.model_config.label_name] = le.transform(df_new['Sentiment'])
    return df_new

def save_model(model):
    # Prepare versioned save file name
    save_file_name = f"{config.app_config.model_save_file}{_version}"
    #TRAINED_MODEL_DIR = "trained_models"
    #save_path = TRAINED_MODEL_DIR / save_file_name
    #save_path = "trained_models/" + save_file_name
    #save_path = "keras.model"
    #model.save(save_path)
    #print("save_model::",save_path)
    #tf.keras.saving.save_model(
    #model, save_path, overwrite=True, save_format=None)
    model.save("sentimental_model.keras")

def load_model(file_name):
    loaded_model = tf.keras.saving.load_model(file_name)
    return loaded_model

def get_tokenizer():
    return tokenizer_ref
    