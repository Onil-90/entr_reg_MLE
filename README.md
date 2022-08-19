# Entropy regularized maximum likelihood estimation for behavior cloning

### How it works 

This is a simple tool that allows to estimate the policy of an agent acting in a discrete (and small) [Markov Decision Process](https://en.wikipedia.org/wiki/Markov_decision_process) from a set of trajectories.
The algorithm recovers the policy by maximizing (via adam) the maximum likelihood with an entropy regularization term. Note that the otpimization here is done in the full policy space, namely the number of parameters is $|\mathcal{S} \times \mathcal{A}|$, where $\mathcal{S}$ is the state space and $\mathcal{A}$ is the action space. In a future I would like to add a function approximation option to allow recovering the policy in large MDPs. 

The function `generate_data()` creates a random policy and sample some trajectory from it. You can specify the size of the MDP (namely the number of states `n_states` and the number of actions `n_actions`) and the amount of trajectories and their length (`n_trajectories` and `trajectory_length`). 

Both the actual policy and the estimated policy will be printed as well as their [KL-divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (more precisely, the Euclidean norm of the vector of the state-wise KL-divergences).


### Requirements

You need [PyTorch](https://pytorch.org/) and [NumPy](https://numpy.org/) to be able to run the code. 
