"""
Entropy regularized maximum likelihood estimation
for recovering the policy of an agent acting in
a discrete environment from a collection of trajectories.
"""

import numpy as np

# Set size of environment
n_states = 81*5
n_actions =  5
# Set regularization coefficient
entr_coeff = .3
# Set seed
np.random.seed(0)
# Number and length of trajectories
n_trajectories = 1000
trajectory_length = 200


def soft_max(tensor, temp=np.ones([n_states,1]), entr=1):
    temp = temp*entr
    exp = np.exp((1/temp)*(tensor - tensor.max(-1)[:,None]))
    Z = exp.sum(-1)[:,None]
    return exp/Z

def generate_policy():
    policy = np.random.standard_normal(size=[n_states,n_actions])
    policy = soft_max(policy)
    return policy

def generate_data(n_trajectories=n_trajectories, trajectory_length=trajectory_length):
    policy = generate_policy()
    list_trajectories = []
    for _ in range(n_trajectories):
        traj = []
        for _ in range(trajectory_length):
            state = np.random.choice(n_states)
            action = np.random.choice(n_actions, p=policy[state,:])
            traj.append([state,action])
        list_trajectories.append(traj)
    return policy, list_trajectories


def entr_reg_MLE(data, entr_coeff=entr_coeff):
    estimated_policy = np.zeros([n_states,n_actions])
    for traj in data:
        for [s,a] in traj:
            estimated_policy[s,a] += 1
    temp = estimated_policy.sum(-1)[:,None]
    print(estimated_policy)
    estimated_policy = soft_max(estimated_policy, temp=temp, entr=entr_coeff)
    return estimated_policy

def KL_div(policy_1, policy_2):
    KL = (policy_2*(np.log(policy_1) - np.log(policy_2))).sum(-1)
    return KL

def main():
    policy, data = generate_data()
    estimated_policy = entr_reg_MLE(data)
    print("Actual policy\n", policy)
    print("Estimated policy\n", estimated_policy.round(4))
    print("KL\n", np.linalg.norm(KL_div(estimated_policy, policy),2))


main()
