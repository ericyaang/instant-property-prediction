import json
import joblib
from pathlib import Path
from config import config


def load_dict(filepath):
    """Load a dictionary from a JSON's filepath."""
    with open(filepath, "r") as fp:
        d = json.load(fp)
    return d


def save_models(pipe_dict, X, y):
    for pipe_name, pipe_info in pipe_dict.items():
        pipe_info.fit(X, y)
    else:
        joblib.dump(pipe_name, Path(config.MODEL_REGISTRY, f"{pipe_name}.bin"))
