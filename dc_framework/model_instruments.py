import logging
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from .data_preparation import Dataset
from .device import use_gpu

logger = logging.getLogger("__name__")

def init(model: torch.nn.Module, criterion: torch.nn.Module):
    return DCFramework(model, criterion)


class DCFramework:
    def __init__(self, model: torch.nn.Module, criterion: torch.nn.Module, lr=1e-2):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.criterion = criterion
        self.device = "cpu"

    def forward(self, feature):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        return output
        

    @use_gpu # this should probably wrap a train_loop function
    def train(self, train_data: Dict[str, np.array], num_epochs : int = 500, batch_size: int = 2, gpu : bool = True, verbose : bool = False, eps : float = 10e-6):
        train_data = Dataset(train_data)
        gpu = ((self.device != "cpu") & gpu)
        # print(f"gpu {gpu}")
        train_dataloader = train_data.get_dataloader(batch_size=batch_size, gpu=gpu)
        #the upper part should move to train_loop.py, here we'll only have a train_dataloader object generated elsewhere
        train_losses = []

        for epoch in range(num_epochs):

            running_loss = 0

            for x_batch, y_batch in train_dataloader:
                if gpu:
                    x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device,non_blocking=True)
                # print(x_batch.is_cuda)
                self.optimizer.zero_grad()
                pred = self.forward(x_batch)
                loss = self.criterion(pred, y_batch)

                running_loss += 1 / (epoch + 1) * (loss.detach().cpu() - running_loss) #check the formula
                
                loss.backward()
                self.optimizer.step()

            train_losses.append(running_loss)
            if verbose:
                print(f"epoch {epoch} train loss is {running_loss}")
            #this should be returned as an epoch loss to train_loop
        return train_losses
    @use_gpu
    def validate(self, val_data: Dict[str, np.array], batch_size: int = 2, gpu=True):
        val_data = Dataset(val_data)
        gpu = ((self.device != "cpu") & gpu)
        # print(f"gpu {gpu}")
        val_dataloader = val_data.get_dataloader(batch_size=batch_size, gpu=gpu)
        val_losses = []
        self.model.eval()
        # print(f'model device {self.device}')
        preds = []
        with torch.no_grad():

            for x_batch, y_batch in val_dataloader:

                if gpu:
                    x_batch, y_batch = x_batch.to(self.device, non_blocking=True), y_batch.to(self.device,non_blocking=True)

                pred = self.forward(x_batch)
                preds.append(pred)
                loss = self.criterion(pred, y_batch)
                val_losses.append(loss)

        return preds, np.mean(np.array(val_losses))


    def dict_save(self, path: Path):
        # state = {
        #     "model": self.model.state_dict(),
        #     "optimizer": self.optimizer.state_dict(),

        # }
        state = self.model.state_dict()
        torch.save(state, path)

    def save_full(self, path: Path):
        model=self.model
        print(model)
        torch.save(model, path)


    def dict_load(self, path):
        self.model.load_state_dict(torch.load(path))


def load_full(path: Path):
    model = torch.load("path")
    return dc_framework.init(model)


