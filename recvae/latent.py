import os
import numpy as np
import pandas as pd

import torch
from torch import optim

import random
from copy import deepcopy

from utils import get_data, ndcg, recall, implicit_slim, load_tr_te_data
from model import VAE

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str)
parser.add_argument('--infer_data', type=str)
parser.add_argument('--hidden-dim', type=int, default=600)
parser.add_argument('--latent-dim', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=500)
parser.add_argument('--model_path', type=str, default="model.pt")

args = parser.parse_args()

seed = 1337
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device("cuda:0")

unique_sid = list()
with open(os.path.join(args.dataset, 'unique_sid.txt'), 'r') as f:
    for line in f:
        unique_sid.append(line.strip())

unique_uid = list()
with open(os.path.join(args.dataset, 'unique_uid.txt'), 'r') as f:
    for line in f:
        unique_uid.append(line.strip())
        
n_items = len(unique_sid)
n_users = len(unique_uid)

data = get_data(args.dataset)
train_data, valid_in_data, valid_out_data, test_in_data, test_out_data = data
infer_data_tr, infer_data_te = load_tr_te_data(args.infer_data,
                                               args.infer_data,
                                               n_items, n_users, 
                                               global_indexing=False)

def generate(batch_size, device, data_in, data_out=None, shuffle=False, samples_perc_per_epoch=1):
    assert 0 < samples_perc_per_epoch <= 1
    
    total_samples = data_in.shape[0]
    samples_per_epoch = int(total_samples * samples_perc_per_epoch)
    
    if shuffle:
        idxlist = np.arange(total_samples)
        np.random.shuffle(idxlist)
        idxlist = idxlist[:samples_per_epoch]
    else:
        idxlist = np.arange(samples_per_epoch)
    
    for st_idx in range(0, samples_per_epoch, batch_size):
        end_idx = min(st_idx + batch_size, samples_per_epoch)
        idx = idxlist[st_idx:end_idx]

        yield Batch(device, idx, data_in, data_out)


class Batch:
    def __init__(self, device, idx, data_in, data_out=None):
        self._device = device
        self._idx = idx
        self._data_in = data_in
        self._data_out = data_out
    
    def get_idx(self):
        return self._idx
    
    def get_idx_to_dev(self):
        return torch.LongTensor(self.get_idx()).to(self._device)
        
    def get_ratings(self, is_out=False):
        data = self._data_out if is_out else self._data_in
        return data[self._idx]
    
    def get_ratings_to_dev(self, is_out=False):
        return torch.Tensor(
            self.get_ratings(is_out).toarray()
        ).to(self._device)


def infer(model, data_in, data_out, samples_perc_per_epoch=1, batch_size=500):
    model.eval()
    result = []
    
    for batch in generate(batch_size=batch_size,
                          device=device,
                          data_in=data_in,
                          data_out=data_out,
                          samples_perc_per_epoch=samples_perc_per_epoch
                         ):
        
        ratings_in = batch.get_ratings_to_dev()
    
        ratings_pred = model(ratings_in, calculate_loss=False).cpu().detach().numpy()
        
        if not (data_in is data_out):
            ratings_pred[batch.get_ratings().nonzero()] = -np.inf

        result.append(ratings_pred)

    result = np.concatenate(result)
    print(result.shape)
        
    return result


model_kwargs = {
    'hidden_dim': args.hidden_dim,
    'latent_dim': args.latent_dim,
    'input_dim': train_data.shape[1]
}
metrics = [{'metric': ndcg, 'k': 100}]

best_ndcg = -np.inf
train_scores, valid_scores = [], []

model = VAE(**model_kwargs).to(device)

model.load_state_dict(torch.load(args.model_path, weights_only=True))
model.to(device)
model.eval()

result = infer(model, infer_data_tr, infer_data_te, 1)

import pickle
with open(f"{args.dataset}/result_csp.pkl", "wb") as f:
    pickle.dump(result, f)
