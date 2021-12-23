import os
import re
from main import X, user_input_features, predict

# Run "python -m pytest test.py" après avoir fait "python -m pip install pytest"
# pour éviter les problèmes de versions de sklearn


def test_model_existence():
    assert os.path.exists("finalized_model.sav") == 1


def test_number_features():
    assert X.shape[1] == user_input_features().shape[1]


def test_prediction_format():
    assert re.match(r"(?:[$]\d+,\d{3})", predict(X.sample()))


