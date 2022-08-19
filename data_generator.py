import numpy as np
import torch
from utils import *


def generate_policy(n_states:int, n_actions:int) -> np.ndarray:
    policy = torch.randn(size=[n_states,n_actions])
    policy = soft_max(policy)
    return policy.numpy()


def generate_data(
        n_states:int,
        n_actions:int,
        n_trajectories:int, 
        trajectory_length:int
        )-> tuple[np.ndarray, list]:
    policy = generate_policy(n_states, n_actions)
    list_trajectories = []
    for _ in range(n_trajectories):
        traj = []
        for _ in range(trajectory_length):
            state = np.random.choice(n_states)
            action = np.random.choice(n_actions, p=policy[state,:])
            traj.append([state,action])
        list_trajectories.append(traj)
    return policy, list_trajectories
