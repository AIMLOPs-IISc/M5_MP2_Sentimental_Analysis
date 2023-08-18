import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pytest
import warnings
warnings.filterwarnings("ignore")
from sentimental_model.processing.data_manager import load_dataset

from sentimental_model.config.core import config

@pytest.fixture
def sample_input_data():
    test_data = load_dataset(file_name = config.model_config.data_file_name)

    return test_data