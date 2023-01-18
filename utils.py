import torch
from torch.utils.data import DataLoader

from membership_functions import *

import json

from functools import partial
from typing import Callable, Iterator

from anfis import ANFIS
from dataset_creation import *
from config import DATASET_PATH, column_names, RANDOM_STATE
from sepsis_dataset import SepsisDataset

from sklearn.model_selection import KFold

from train import train
from test import run_inference

def generate_membership_functions_from_json (path2json : str, columns : list):
    
    membership_funcs = dict()
    with open(path2json, 'r') as f:
        file_dict = json.load(f)

        for class_name, class_items in file_dict.items():
            for fuzzy_name, fuzzy_items in class_items.items():
                membership_func_class = eval(fuzzy_items['membership_function'])
                membership_func = partial(
                    membership_func_class,
                    **fuzzy_items['params'],
                    feature_idx = columns.index(class_name)
                )
                membership_funcs[f"{class_name}_{fuzzy_name}"] = membership_func


    return membership_funcs

def generate_rules_object_from_file (path2rulef : str, verbose : bool = False):

    rules = []
    with open(path2rulef, 'r') as f:
        for line in f.readlines():
            line = line.strip()
            rules_str, out, is_skip = line.split(' -> ')

            if is_skip == 'SKIP':
                if verbose: print(f"Skipping: {line}")
                continue
            
            class_val = 0 if out == "NO SEPSIS" else 1
            rules.append(
                (rules_str.split(" AND "), class_val)
            )

    return rules

def cross_validate (
    model_constructor : Callable[[], ANFIS],
    train_data : pd.DataFrame,
    batch_size : int,
    criterion : Callable[[], nn.modules.loss._Loss],
    optimizer : Callable[[Iterator[nn.parameter.Parameter]], torch.optim.Optimizer],
    epoch_count : int,
    fold_amount : int = 5
) -> ANFIS:

    kf = KFold(n_splits=fold_amount, shuffle=True, random_state=RANDOM_STATE)

    score_dict = dict()
    for fold_idx, (train_index, validation_index) in enumerate(kf.split(train_data)):
        
        train_dataset = SepsisDataset(train_data.iloc[train_index])
        val_dataset = SepsisDataset(train_data.iloc[validation_index])

        model = train(
            model_constructor,
            train_dataset=train_dataset,
            batch_size=batch_size,
            criterion=criterion,
            optimizer=optimizer,
            epoch_count=epoch_count,
            phase_name=f"fold_{fold_idx}_train"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=len(train_dataset)
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset)
        )
        fold_scores_train = run_inference(
            model,
            train_loader,
            phase=f"fold_{fold_idx}_training"
        )
        fold_scores_val = run_inference(
            model,
            val_loader,
            phase=f"fold_{fold_idx}_validation"
        )

        score_dict[f"fold_{fold_idx+1}"] = {
            "train" : fold_scores_train,
            "validation" : fold_scores_val
        }

    return score_dict

        

        



if __name__ == "__main__":
    path2json = "membership_functions.json"
    path2rulef = "rules.txt"
    data = prepare_data(DATASET_PATH, column_names).iloc[:500]
    membership_funcs = generate_membership_functions_from_json(path2json, data.columns.to_list())
    

    rules = generate_rules_object_from_file(path2rulef, True)
    model = ANFIS(
        membership_funcs,
        rules,
        feature_num=7,
        class_num=2
    )

