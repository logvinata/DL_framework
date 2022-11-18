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

    def forward(self, feature, target):
        try:
            output = self.model(feature)
        except:
            logger.warning(f"feature: {feature}")
            raise
        try:
            loss = self.criterion(output, target)
        except:
            logger.warning(f"output: {output}")
            logger.warning(f"target: {target}")
            raise
        return {
            "output": output,
            "loss": loss
        }
    @use_gpu # this should probably wrap a train_loop function
    def train(self, train_data: Dict[str, np.array], batch_size: int = 1, gpu : bool = True):
        train_data = Dataset(train_data)
        train_dataloader = train_data.get_dataloader(batch_size=batch_size, gpu=gpu)
        #the upper part should move to train_loop.py, here we'll only have a train_dataloader object generated elsewhere
        train_losses = []

        for batch in train_dataloader:
            self.optimizer.zero_grad()
            output = self.forward(*batch)
            loss = output["loss"]
            train_losses.append(loss.item())
            loss.backward()
            self.optimizer.step()

        print(f"train loss is {np.mean(np.array(train_losses))}")
        #this should be returned as an epoch loss to train_loop

    def validate(self, val_data: Dict[str, np.array], batch_size: int = 1, gpu=True):
        val_data = Dataset(val_data)
        val_dataloader = val_data.get_dataloader(batch_size=batch_size, gpu=gpu)
        val_losses = []
        self.model.eval()
        preds = []
        for batch in val_dataloader:
            val_output = self.forward(*batch)
            val_losses.append(val_output["loss"].item())
            preds.append(val_output["output"])


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


