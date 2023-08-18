import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sentimental_model import __version__ as _version
from sentimental_model.config.core import config
from sentimental_model.processing.data_manager import load_dataset

def run_training():
    test_data = load_dataset(file_name = config.model_config.data_file_name)
    print("test_data::",len(test_data))


if __name__ == "__main__":
    run_training()
    