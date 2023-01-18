import torch
import torch.nn as nn

from config import *
from dataset_creation import prepare_data

from sepsis_dataset import SepsisDataset

from train import train

from anfis import ANFIS

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from functools import partial

from utils import (
    generate_rules_object_from_file,
    generate_membership_functions_from_json,
    cross_validate
)

torch.manual_seed(RANDOM_STATE)

if __name__ == "__main__":

    dataframe = prepare_data(DATASET_PATH, column_names)

    # replace nan values with mean in the training set
    X_train, X_test = train_test_split(dataframe, random_state=RANDOM_STATE)
    X_train.fillna(X_train.mean(), inplace=True)

    train_dataset = SepsisDataset(X_train)
    test_dataset = SepsisDataset(X_test)

    batch_size = 128
    epoch_count = 10

    PHASE = "train"

    rules = generate_rules_object_from_file(RULE_PATH)
    membership_functions = generate_membership_functions_from_json(
        MEMBERSHIP_FUNC_PATH,
        dataframe.columns.to_list()
    )
    model = partial(
        ANFIS,
        rules = rules,
        membership_functions = membership_functions,
        feature_num = len(dataframe.columns) - 1,
        class_num = 2
    )

    criterion = partial(nn.CrossEntropyLoss, )

    optimizer = partial(
        torch.optim.Adam,
        lr = 1e-3
    )

    score_dict = cross_validate(
        model,
        train_data=X_train,
        batch_size=batch_size,
        criterion=criterion,
        optimizer=optimizer,
        epoch_count=epoch_count,
        fold_amount=5
    )
    