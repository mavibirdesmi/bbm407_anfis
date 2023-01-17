import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np

class SepsisDataset (Dataset):
    def __init__ (
        self,
        dataframe : pd.DataFrame
    ):

        self.X = torch.tensor(
            dataframe.drop(['sepsis_icd'])
        )
        self.y = torch.tensor(
            dataframe['sepsis_icd']
        )

    def __len__ (self):
        return len(self.X)
    
    def __getitem__ (self, idx) :
        return self.X[idx], self.y[idx]