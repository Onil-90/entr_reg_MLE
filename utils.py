import torch
import numpy as np


def soft_max(tensor:torch.Tensor, temperature:float=1) -> torch.Tensor:
    exp = torch.exp((1/temperature)*(tensor))
    Z = exp.sum(-1)
    Z = torch.unsqueeze(Z,-1)
    return exp/Z


def KL_div(policy_1:np.ndarray, policy_2:np.ndarray) -> np.ndarray:
    KL = (policy_2*(np.log(policy_1) - np.log(policy_2))).sum(-1)
    return KL
