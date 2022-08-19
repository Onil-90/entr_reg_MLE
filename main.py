"""
Entropy regularized maximum likelihood estimation
for recovering the policy of an agent acting in
a discrete environment from a collection of trajectories.
"""

from data_generator import generate_data
from mle import maximum_likelihood
from utils import *
import numpy as np

# Set size of environment
n_states = 10
n_actions =  4
# Set regularization coefficient
entr_coeff = .3
# Set number of epochs and learning rate
n_epochs = 10
lr = 0.1
# Set seed
np.random.seed(0)
# Number and length of trajectories
n_trajectories = 30
trajectory_length = 10


def main():
    """Generate a policy and a list of trajectories"""
    policy, data = generate_data(
        n_states = n_states, 
        n_actions = n_actions, 
        n_trajectories=n_trajectories,
        trajectory_length=trajectory_length)
    """Recover the policy from the trajectories"""
    estimated_policy = maximum_likelihood(
        n_states=n_states,
        n_actions=n_actions,
        tjs_list = data,
        n_epochs = n_epochs,
        lr = lr,
        entr_coeff = entr_coeff
    )
    print("Actual policy\n", policy)
    print("Estimated policy\n", estimated_policy.round(4))
    print("KL divergence between the estimated policy and the actual policy>\n",
        np.linalg.norm(KL_div(estimated_policy, policy),2))


main()
