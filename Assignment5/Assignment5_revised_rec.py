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

def get_prob_hyperspeed(bool_dict, node_value):
    if type(node_value) is float:
        return node_value
    else:
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

# Initiate total weights
W_all = 0
W_strict = 0

#%%
# Create the dataset
info = {
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
count_all = 0
count_strict = 0
for i in range(N):
    tmp = get_sample(info)
    if tmp['B'] and tmp['L']:
        count_all += 1
        if tmp['D']:
            count_strict += 1
p_d_bl = round(count_strict/count_all, 3)

#%%
print(p_d_bl)

#%%
count_all = 0
count_strict = 0
for i in range(N):
    tmp = get_sample(info)
    if tmp['V']:
        count_all += 1
        if tmp['X']:
            count_strict += 1
p_x_v = round(count_strict/count_all, 3)

#%%
print(p_x_v)

#%%
count_all = 0
count_strict = 0
for i in range(N):
    tmp = get_sample(info)
    if (not tmp['V']) and tmp['S']:
        count_all += 1
        if tmp['T'] or tmp['L']:
            count_strict += 1
p_t_or_l_vc_s = round(count_strict/count_all, 3)
#%%
print(p_t_or_l_vc_s)

#%% md

# ## II. Likelihood sampling

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
#%%
W_all = 0
W_strict = 0
for i in range(N):
    tmp, w = get_sample_mlw(info, cond={'B':True})
    if tmp['B'] and tmp['L']:
        W_all += w
        if tmp['D']:
            W_strict += w
p_d_b = W_strict / W_all

#%%
print(p_d_b)

#%%
W_all = 0
W_strict = 0
for i in range(N):
    tmp, w = get_sample_mlw(info, cond={'V':True})
    if tmp['V']:
        W_all += w
        if tmp['X']:
            W_strict += w
p_x_v = W_strict / W_all
#%%
print(p_x_v)

#%%
W_all = 0
W_strict = 0
for i in range(N):
    tmp, w = get_sample_mlw(info, cond={'V':False, 'S':True})
    if (not tmp['V']) and tmp['S']:
        W_all += w
        if tmp['T'] or tmp['L']:
            W_strict += w
p_t_or_l_vc_s = W_strict / W_all
#%%
print(p_t_or_l_vc_s)
