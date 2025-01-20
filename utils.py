import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
# from time import time
# from model import LightGCN
# from model import PairWiseModel
from sklearn.metrics import roc_auc_score
import random
import os
try:
    from cppimport import imp_from_filepath
    from os.path import join, dirname
    path = join(dirname(__file__), "sources/sampling.cpp")
    sampling = imp_from_filepath(path)
    sampling.seed(world.seed)
    sample_ext = True
except:
    print("Cpp extension not loaded")
    sample_ext = False


def set_seed(seed):
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

def getFileName():
    # if 'mf' in world.model_name:
    #     file = f"{world.config['log_file'][6:]}.pth.tar"
    # elif 'lgn' in world.model_name:
    #     file = f"{world.config['log_file'][6:]}.pth.tar"
    if world.LOAD == 1:
        file = f"{world.config['log_file'][11:]}.pth.tar"
    else:
        file = f"{world.config['log_file'][6:]}.pth.tar"
    return os.path.join(world.FILE_PATH,file)

def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['bpr_batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)