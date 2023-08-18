"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf
import pandas as pd
#from sentimental_model import __version__ as _version
from sentimental_model.config.core import config

from sentimental_model.processing.data_manager import load_model, get_tokenizer,load_dataset,split_data,convert_text_to_int_values,cleanText,remove_stopwords

from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict(test_samples):
    

    samp = pd.DataFrame(test_samples,columns = ['Text'])
    samp['Text'] = samp['Text'].apply(lambda a :  remove_stopwords(cleanText(a)))
    
    sample_arr = np.array(samp['Text'])
    
    model = load_model("sentimental_model.keras")
    
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(sample_arr)
 
    # YOUR CODE HERE
    test_samples_pro = pad_sequences(
        tokenizer.texts_to_sequences(sample_arr), 
        padding=config.model_config.padding_val, 
        maxlen=config.model_config.maxlen, 
        truncating=config.model_config.truncating_val)

    # predict
    pred = model.predict(x=test_samples_pro)
    print ("summ:")
    print(model.summary())
    print("pred::",pred)

if __name__ == "__main__":
    test_sample_1 = "This is good product"
    test_sample_2 = "This is waste product"
       
    test_samples = [ test_sample_1, test_sample_2]
    
    predict(test_samples)
    
