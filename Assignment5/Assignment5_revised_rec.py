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
import pandas as pd
import numpy as np
from random import random
from pandas import DataFrame as fr

# Given dictionary (dict) and dataframe (df). Ouput the subset of the dict that contains only keys that are also in the df
def subset_dict(dictionary, df):
    tmp_dict = {}
    for x in dictionary:
        if x in df.keys():
            tmp_dict[x] = dictionary[x]
    return tmp_dict

# Subset the dataframe (df), so it meets all the conditions set by the boolean dictionary (bool_dict). Output the probability associated with the subsetted df
def get_prob(bool_dict, df):
    # Ensure that we only set conditions on keys
    # that exist in the dataframe
    s = subset_dict(bool_dict, df)
    qry = ' and '.join(["{} == {}".format(k,v) for k,v in s.items()])
    if qry != "":
        return df.query(qry).prob.iloc[0]
    else:
        return df.prob.iloc[0]

def get_sample(graph, bool_dict={}):
    for key in graph:
        r = random()
        prob = get_prob(bool_dict, graph[key])
        bool_dict[key] = (prob > r)
    return bool_dict

# Initiate total weights
W_all = 0
W_strict = 0

#%%
# Create the dataset
tf2 = [True, False]
tf4_1 = [True, True, False, False]
tf4_2 = [True, False, True, False]
info = {
'V': fr({'prob': [0.01]}),                                       # 1st row
'S': fr({'prob': [0.5]}),
'T': fr({'prob': [0.05, 0.01],'V': tf2}),                        # 2nd row
'L': fr({'prob': [0.1, 0.01], 'S': tf2}),
'B': fr({'prob': [0.6, 0.3], 'S': tf2}),
'C': fr({'prob': [1.0, 1.0, 1.0, 0.0], 'T': tf4_1, 'L': tf4_2}), # 3rd row
'X': fr({'prob': [0.98, 0.05], 'C': tf2}),                       # 4th row
'D': fr({'prob': [0.9, 0.7, 0.8, 0.9], 'C': tf4_1, 'B': tf4_2})
}

#%%
N = 1000
count_all = 0
count_strict = 0
for i in range(N):
    tmp = get_sample(info)
    if tmp['B'] and tmp['C']:
        count_all += 1
        if tmp['D']:
            count_strict += 1
p_d_bc = round(count_strict/count_all, 3)

#%%
print(p_d_bc)

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
        if tmp['C']:
            count_strict += 1
p_vc_s = round(count_strict/count_all, 3)
#%%
print(p_vc_s)

#%%
# For a graph
def get_sample_mlw(graph, bool_dict={}, cond={}):
    # Initiate the weight of the sample
    w = 1
    for key in graph:
        r = random()
        if key in cond:
            bool_dict[key] = cond[key]
            w *= get_prob(bool_dict, graph[key])
        else:
            prob = get_prob(bool_dict, graph[key])
            bool_dict[key] = (prob > r)
    return [bool_dict, w]
#%%


#%%
N = 1000
count_all = 0
count_strict = 0
for i in range(N):
    tmp = get_sample(info)
    if tmp['B'] and tmp['C']:
        count_all += 1
        if tmp['D']:
            count_strict += 1
p_d_bc = round(count_strict/count_all, 3)

W_all = 0
W_strict = 0
for i in range(N):
    tmp, w = get_sample_mlw(info, cond={'B':True, 'C':True})
    if tmp['B'] and tmp['C']:
        W_all += w
        if tmp['D']:
            W_strict += w
p_d_bc = W_strict / W_all

def p_d_bc(info, N, get_sample_fun):
    count_all = 0
    count_strict = 0
    for i in range(N):
        tmp = get_sample(info)
        if tmp['B'] and tmp['C']:
            count_all += 1
            if tmp['D']:
                count_strict += 1
    p_d_bc = round(count_strict/count_all, 3)



#%%
print(round(W_strict / W_all , 2))

# Test by generating all probabilities of directed graph
