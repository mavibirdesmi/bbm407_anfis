import os

DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "ebru_hoca_sepsis_dataset-2/ebru_hoca_dataset/")
FILE_NAME_START = 500
FILE_NAME_OUT = 1000
RANDOM_STATE = 42

column_names = [
    "user_id",
    "heart_rate",
    "bp_systolic",
    "bp_diastolic",
    "map",
    "resp",
    "temp",
    "spo2",
    "fio2",
    "wbc",
    "bun",
    "bilirubin",
    "creatinine",
    "lactate",
    "platelets",
    "ph",
    "pco2",
    "po2",
    "bicarbonate",
    "hemoglobin",
    "hematocrit",
    "potassium",
    "chloride",
    "gcs",
    "age",
    "sepsis_icd",
    "sirs",
    "qsofa"
]