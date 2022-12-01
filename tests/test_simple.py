# import sys
# sys.path.insert(0, '../')

import numpy as np

import torch
import dc_framework

import matplotlib.pyplot as plt


def train_simple_model():
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        # torch.nn.Sigmoid()
    )
    criterion = torch.nn.MSELoss()

    train_data = {
        "feature": np.array([[0, 0], [0, 1], [1, 0], [1, 1], [3, 3], [4, 4]]),
        "target": np.array([0, 0.3, 0.5, 0.8, 2.4, 3.2])
    }

    val_data = {
    "feature": np.array([[0, 0], [0, 2], [2, 0], [2, 2]]),
    "target": np.array([0, 0.6, 1.1, 1.6])
    }

    batch_size = 4
    num_epochs = 100

    model_dc_framework = dc_framework.init(model, criterion)
    losses = model_dc_framework.train(train_data, num_epochs, batch_size, gpu = False, verbose=True)  # pass gpu=False if you don't want gpu
    print (f'loss = {losses[-1]}')
    
    # model_dc_framework.dict_save("model_dict")
    # model_dc_framework.save_full("full_model")
    
    preds, val_loss = model_dc_framework.validate(val_data=val_data, batch_size=batch_size)
    print(f"validation loss is {val_loss}")
    print(f"Predictions {preds}")

def load_model_dict_test(path, validate=True):
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 1),
        # torch.nn.Sigmoid()
    )
    criterion = torch.nn.MSELoss()
    model_from_dict = dc_framework.init(model, criterion)
    # print(model_from_dict.model)
    model_from_dict.dict_load(path)
    if validate:
        val_data = {
        "feature": np.array([[0, 0], [0, 2], [2, 0], [2, 2]]),
        "target": np.array([0, 0.6, 1.1, 1.6])
        }
        preds, val_loss = model_from_dict.validate(val_data)
        print(f"validation loss is {val_loss}")
        print(f"Predictions {preds}")





if __name__ == "__main__":
    train_simple_model()
    # load_model_dict_test("model_dict")

