import torch
import torch.nn as nn

from config import *
from dataset_creation import prepare_data

from sepsis_dataset import SepsisDataset

from train import train

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

if __name__ == "__main__":

    dataframe = prepare_data(DATASET_PATH, column_names)

    # replace nan values with mean in the training set
    X_train, X_test = train_test_split(dataframe, random_state=RANDOM_STATE)
    X_train.fillna(X_train.mean())

    train_dataset = SepsisDataset(X_train)
    test_dataset = SepsisDataset(X_test)

    batch_size = 128

    PHASE = "train"

    train(
        
    )
    