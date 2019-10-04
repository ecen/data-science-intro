#%% md
# # Assignment 5
# ## Students:
- Eric Guldbrand
- Davíð Freyr Björnsson
- Time spent:

#%% md
# ### Three different resampling algorithms:

1. Rejection sampling
2. Likelihood weighted sampling
3. Gibbs sampling

#%% md
# ### Apply each of the three algorithms to the following tasks.
# ### 1. Compute the following probabilities, directly, without sampling. Then employ each of the three sampling algorithms to approximate the probabilities. Use 1000 samples for each method and document your results. How do the approximations compare to the true values?

# ### a. $P(D | B,C) = P(D = true) | B = true, C = true)$
Solution: $P(D = true) | B = true, C = true) = 0.9$ (from table)

# ### b. $P(X | V) = P(X = true | V = true)$
Solution: $P(X = true | V = true) = \frac{P(X = true, V = true)}{P(V = true)}$. But:
$$P(X = true, V = true) = $$

#%% md

# ### c. $P(C | V^c , S) = P(C = true | V = false, S = true)$

#%%
import numpy as np
from random import random

# Get the probability of an event happening (node)
# given a dictionary of conditions (bool_dict)
def get_prob_hyperspeed(bool_dict, node_value):
    # No conditions
    if type(node_value) is float:
        return node_value
    else:
    # One or more conditions
        dict = node_value[1]
        for label in node_value[0]:
            dict = dict[bool_dict[label]]
        return dict

def get_sample(graph, bool_dict={}):
    for key in graph:
        r = random()
        prob = get_prob_hyperspeed(bool_dict, graph[key])
        bool_dict[key] = (prob > r)
    return bool_dict

# Evaluate a condition on the graph using rejection sampling
# inner, the list of nodes we want the probability of being true: ['A', 'B']
# conditions, dict of node states that are given: {'C': True, 'D': False}
# N, number of samples to use
def rejection_sample(inner, conditions, N):
    count_all = 0
    count_strict = 0
    for i in range(N):
        bool_dict = get_sample(graph)
        if and_conditions(conditions, bool_dict):
            count_all += 1
            if or_nodes(inner, bool_dict):
                count_strict += 1
    prob = count_strict / count_all
    return prob

def and_conditions(conditions, boolean_dict):
    for name in conditions.keys():
        if conditions[name] != boolean_dict[name]:
            return False
    return True

def or_nodes(nodeNames, boolean_dict):
    for name in nodeNames:
        if boolean_dict[name]:
            return True
    return False

# Initiate total weights
W_all = 0
W_strict = 0

#%%
# Create the dataset
graph = {
'V': 0.01,                                       # 1st row
'S': 0.5,
'T': (['V'], {True: 0.05, False: 0.01}),                        # 2nd row
'L': (['S'], {True: 0.1, False: 0.01}),
'B': (['S'], {True: 0.6, False: 0.3}),
'X': (['T', 'L'], {True: {True: 0.98, False: 0.98}, False: {True: 0.98, False: 0.05}}), # 3rd row
'D': (['T', 'L', 'B'], {True: {True: {True: 0.9, False: 0.7}, False: {True: 0.9, False: 0.7}}, False: {True: {True: 0.9, False: 0.7}, False: {True: 0.8, False: 0.9}}}),
}

#%% md

# ## I. Rejection sampling

#%%
N = 1000000

p_d_bl = rejection_sample(['D'], {'B': True, 'L': True}, N)
print(round(p_d_bl, 3))

#%%
p_x_v = rejection_sample(['X'], {'V': True}, N)
print(round(p_x_v, 3))

#%%
p_t_or_l_vc_s = rejection_sample(['T', 'L'], {'V': False, 'S': True}, N)
print(round(p_t_or_l_vc_s, 3))

#%% md

# ## II. Likelihood sampling

# ## Define functions for likelihood sampling calculations

#%%
# For a graph
def get_sample_mlw(graph, bool_dict={}, cond={}):
    # Initiate the weight of the sample
    w = 1
    for key in graph:
        r = random()
        if key in cond:
            bool_dict[key] = cond[key]
            w *= get_prob_hyperspeed(bool_dict, graph[key])
        else:
            prob = get_prob_hyperspeed(bool_dict, graph[key])
            bool_dict[key] = (prob > r)
    return [bool_dict, w]

# outer, lists {'V': False, 'S': True}
# inner, ['T', 'L'], the variables we want to have the probability of
def likelihood_sample(inner, conditions, N):
    W_all = 0
    W_strict = 0
    for i in range(N):
        bool_dict, w = get_sample_mlw(graph, cond=conditions)
        if and_conditions(conditions, bool_dict):
            W_all += w
            if or_nodes(inner, bool_dict):
                W_strict += w
    prob = W_strict / W_all
    return prob
#%%
p_d_b = likelihood_sample(['D'], {'B': True, 'L': True}, N)
print(round(p_d_b, 3))

#%%
p_x_v = likelihood_sample(['X'], {'V': True}, N)
print(round(p_x_v, 3))

#%%
p_t_or_l_vc_s = likelihood_sample(['T', 'L'], {'V': False, 'S': True}, N)
print(round(p_t_or_l_vc_s, 3))
