import pandas as pd
import numpy as np
import os

from typing import List

from tqdm import tqdm

from config import DATASET_PATH, column_names

def load_sepsis_data (path:str, column_names : List[str]) -> pd.DataFrame:
    df = pd.DataFrame(columns=column_names)
    
    tbar = tqdm(os.walk(path))
    for root, dirs, files in tbar:
        for file_name in files:
            if not os.path.split(root)[-1].startswith('.') and file_name.endswith(".csv") and int(os.path.splitext(file_name)[0]) in range(200, 501):
                tbar.set_postfix({"file_name" : os.path.join(root,file_name)})
                df_new = pd.read_csv(os.path.join(root,file_name))

                is_sepsis = os.path.split(root)[-1] == 'sepsis'
                prefix = 's' if is_sepsis else 'ns'

                df_new["user_id"] = [f"{prefix}-{os.path.splitext(file_name)[0]}_{i}" for i in range(len(df_new))]
                
                df = pd.concat([df, df_new], ignore_index=True, sort=False)
    
    return df

def remove_columns_with_high_missing_values (dataframe : pd.DataFrame) -> pd.DataFrame:
    for col_name in ["fio2", "bilirubin", "lactate", "ph", "pco2", "po2"]:
        dataframe.drop(col_name, axis=1, inplace=True)
    
    return dataframe

def remove_columns_with_high_correlation (dataframe : pd.DataFrame) -> pd.DataFrame:
    for col_name in ["hematocrit", "bp_diastolic", "map", "bun", "resp"]:
        dataframe.drop(col_name, axis=1, inplace=True)

    return dataframe

def remove_samples_with_no_label (dataframe : pd.DataFrame) -> pd.DataFrame:
    null_mask = dataframe['sepsis_icd'].isnull()

    return dataframe[~null_mask]

def prepare_data (
    root_path : str,
    column_names : List[str]
):
    # load data from file
    dataframe = load_sepsis_data(root_path, column_names)

    # apply transformations
    dataframe = remove_columns_with_high_missing_values(dataframe)
    dataframe = remove_columns_with_high_correlation(dataframe)
    dataframe = remove_samples_with_no_label(dataframe)

    # get the most important features
    dataframe = dataframe[[
        "bicarbonate",
        "heart_rate",
        "creatinine",
        "hemoglobin",
        "gcs",
        "bp_systolic",
        "wbc"
    ]]

    return dataframe



if __name__ == '__main__':


    df = load_sepsis_data(DATASET_PATH, column_names)
    transformations = [
        remove_columns_with_high_correlation,
        remove_columns_with_high_missing_values,
        remove_samples_with_no_label
    ]
    for transform in transformations:
        df = transform(df)

    print(df.isnull().sum())

    with open("data.csv", "w") as file:
        df.to_csv(file, index=False)