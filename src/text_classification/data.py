from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(data_dir):
    data_dir = Path(data_dir)

    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")

    return train, test


def split_indices(train, seed, test_size=0.2):
    return train_test_split(
        np.arange(len(train)),
        test_size=test_size,
        random_state=seed,
        stratify=train["target"],
    )