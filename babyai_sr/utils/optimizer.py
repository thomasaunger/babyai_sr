import os
import torch

import babyai.utils


def get_optimizer_path(model_name):
    return os.path.join(babyai.utils.get_model_dir(model_name), "optimizer.pt")


def load_optimizer(model_name, raise_not_found=True):
    path = get_optimizer_path(model_name)
    try:
        if torch.cuda.is_available():
            optimizer = torch.load(path)
        else:
            optimizer = torch.load(path, map_location=torch.device("cpu"))
        return optimizer
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No optimizer found at {}".format(path))


def save_optimizer(optimizer, model_name):
    path = get_optimizer_path(model_name)
    babyai.utils.create_folders_if_necessary(path)
    torch.save(optimizer, path)
