from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    return load_train_data(data_dir), load_test_data(data_dir)


def load_train_data(data_dir):
    return pd.read_csv(Path(data_dir) / "train.csv")


def load_test_data(data_dir):
    return pd.read_csv(Path(data_dir) / "test.csv")


def split_indices(train, seed, test_size=0.2):
    return train_test_split(
        np.arange(len(train)),
        test_size=test_size,
        random_state=seed,
        stratify=train["target"],
    )
