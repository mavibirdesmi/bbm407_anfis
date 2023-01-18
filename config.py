import os

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(PROJECT_DIR, "..", "data", "ebru_hoca_sepsis_dataset-2/ebru_hoca_dataset/")
RULE_PATH = os.path.join(PROJECT_DIR, "rules.txt")
MEMBERSHIP_FUNC_PATH = os.path.join(PROJECT_DIR, "membership_functions.json")
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