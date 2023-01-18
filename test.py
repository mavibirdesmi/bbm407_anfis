import torch 
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from anfis import ANFIS

def run_inference (
    model : ANFIS,
    data_loader : DataLoader,
    phase : str
):
    all_scores = dict()
    for iter_idx, data in enumerate(data_loader):
        with torch.no_grad():
            input, target = data

            pred = model(input)

            pred_labels = pred.argmax(dim=1)
            for score_function in [
                accuracy_score,
                f1_score,
                precision_score,
                recall_score
            ]:
                score = score_function(pred_labels, target)
                all_scores[score_function.__name__] = score
    
    return all_scores




