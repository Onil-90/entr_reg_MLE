import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
import numpy as np
from utils import *

""" Allow use of GPU if cuda is available"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Entropy regularized maximum likelihood estimation """
def maximum_likelihood(
        n_states:int, 
        n_actions:int, 
        tjs_list:list, 
        n_epochs:int, 
        lr:float, 
        entr_coeff:float
    ) -> np.ndarray:
    params = [nn.Parameter(torch.rand(size=[n_states, n_actions]).to(device), requires_grad=True)]
    optimizer = torch.optim.Adam(params, lr=lr)
    C = torch.zeros([n_states, n_actions])   
    for tjs in tjs_list:
        for [state, action] in tjs:
            C[state, action] += 1
    for epoch in range(n_epochs):
        loss = 0
        dist = Categorical(torch.exp(params[0]))
        log_probs = torch.log(dist.probs)
        entr_vec = dist.entropy()
        loss = -((log_probs*C).sum() + entr_coeff*(entr_vec*C.sum(-1)).sum())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return Categorical(torch.exp(params[0]).cpu()).probs.detach().numpy()