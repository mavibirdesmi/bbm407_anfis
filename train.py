import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from anfis import ANFIS
from sepsis_dataset import SepsisDataset

from typing import Callable, Iterator

from test import run_inference

from tqdm.auto import tqdm

def train_epoch (
    model : ANFIS,
    train_loader : DataLoader,
    epoch_index : int,
    criterion : nn.modules.loss._Loss,
    optimizer : torch.optim.Optimizer,
    prefix : str = ''
):

    running_loss = 0.
    last_loss = 0.

    tbar = tqdm(enumerate(train_loader), position=0, leave=False)
    for iter_idx, data in tbar:
        
        # get features and target
        input, target = data

        # reset grads befor iteration
        optimizer.zero_grad()

        # make prediction
        pred = model(input)

        # calculate loss
        loss = criterion(pred, target)
        loss.backward()

        # update parameters
        optimizer.step()

        running_loss += loss.item()
        if iter_idx % 10 == 9:
            last_loss = running_loss / 10
            tbar.set_description(
                f"{prefix} Batch {epoch_index+1:5} - Iteration {iter_idx+1}/{len(train_loader)}: {last_loss}",
                refresh=True
            )
            tbar.refresh()


    

def train (
    model : Callable[[], ANFIS],
    train_dataset : SepsisDataset,
    batch_size : int,
    criterion : Callable[[], nn.modules.loss._Loss],
    optimizer : Callable[[Iterator[nn.parameter.Parameter]], torch.optim.Optimizer],
    phase_name : str,
    epoch_count : int
) -> ANFIS:

    # prepare_loaders 
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size
    )

    # initialize model
    model = model()

    # prepare criterion and optimizer
    optimizer = optimizer(model.parameters())

    criterion = criterion()

    for epoch_idx in tqdm(range(epoch_count)):
        train_epoch(
            model=model,
            train_loader=train_loader,
            epoch_index=epoch_idx,
            criterion=criterion,
            optimizer=optimizer,
            prefix=phase_name
        )

    return model






    


