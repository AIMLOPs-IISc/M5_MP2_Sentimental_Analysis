"""
Note: These tests will fail if you have not first trained the model.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sentimental_model.config.core import config

from sentimental_model.processing.data_manager import load_dataset, save_model, split_data, convert_text_to_int_values

from sentimental_model.model.model import create_model 
from sentimental_model.model.train_model import trian_model, evaluate_model


def run_training():
    print("Running Train:",config.model_config.batch_size)
    test_data = load_dataset(file_name = config.model_config.data_file_name)
    print("test_data::",len(test_data))
    
    X_train, X_val, X_test,y_train, y_val, y_test = split_data(
        test_data, 
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
    # save model
    save_model(model_1)
    
if __name__ == "__main__":
    run_training()