#%% md
# # Assignment 5
# ## Students:
- Eric Guldbrand
- Davíð Freyr Björnsson
- Time spent: 20 hours.

#%% md
# ### Three different resampling algorithms:

1. Rejection sampling
2. Likelihood weighted sampling
3. Gibbs sampling

#%% md
# ### Apply each of the three algorithms to the following tasks.
# ### 1. Compute the following probabilities, directly, without sampling. Then employ each of the three sampling algorithms to approximate the probabilities. Use 1000 samples for each method and document your results. How do the approximations compare to the true values?

# ### a. $P(D | B,C) = P(D = true) | B = true, C = true)$
# ### b. $P(X | V) = P(X = true | V = true)$
![](1_ab.jpg)

# ### c. $P(C | V^c , S) = P(C = true | V = false, S = true)$
![](1_c.jpg)

#%%
import numpy as np
from random import random, choice

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
    if count_all == 0:
        return 0
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
N = 1000

p_d_bl = rejection_sample(['D'], {'B': True, 'L':  True}, N)
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

# Evaluate a condition on the graph using likelihood weighted sampling
# inner, the list of nodes we want the probability of being true: ['A', 'B']
# conditions, dict of node states that are given: {'C': True, 'D': False}
# N, number of samples to use
def likelihood_sample(inner, conditions, N):
    W_all = 0
    W_strict = 0
    for i in range(N):
        bool_dict, w = get_sample_mlw(graph, cond=conditions)
        if and_conditions(conditions, bool_dict):
            W_all += w
            if or_nodes(inner, bool_dict):
                W_strict += w
    if W_all == 0:
        return 0
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

#%% md
# ## III. Gibbs sampling

#%%
def gibbs_sample(inner, conditions, N, burN):
    bool_dict, w = get_sample_mlw(graph, cond=conditions)
    count_all = count_strict = 0
    keys_set = list(set(bool_dict) - set(conditions))
    for i in range(0, N+burN):
        random_key = choice(keys_set)
        gibbs_conditions = bool_dict.copy()
        gibbs_conditions.pop(random_key, None)
        prob = get_prob_hyperspeed(gibbs_conditions, graph[random_key])
        r = random()
        bool_dict[random_key] = r < prob
        if i >= burN:
            if and_conditions(conditions, bool_dict):
                count_all += 1
                if or_nodes(inner, bool_dict):
                    count_strict += 1
    if count_all == 0:
        return 0
    prob = count_strict / count_all
    return prob

#%%
p_d_b = gibbs_sample(['D'], {'B': True, 'L': True}, N, 0)
print(round(p_d_b, 3))

p_x_v = gibbs_sample(['X'], {'V': True}, N, 0)
print(round(p_x_v, 3))

p_t_or_l_vc_s = gibbs_sample(['T', 'L'], {'V': False, 'S': True}, N, 0)
print(round(p_t_or_l_vc_s, 3))

#%% md
# # 2. Accuracy vs samples
Now focus on the probability in 1a, $P(D | B, C)$ We know that the accuracy of the sampling approximations depends on the number of samples used. For each of the three sampling methods, plot the probability $P(D | B, C)$ as a function of the number of samples used by the sampling method. Is there any difference between the methods?

#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sample_size = range(100, 10000, 10)
rejection = []
mlw = []
gibbs = []
for i in sample_size:
    rejection.append(rejection_sample(['D'], {'B': True, 'L': True}, i))
    mlw.append(likelihood_sample(['D'], {'B': True, 'L': True}, i))
    gibbs.append(gibbs_sample(['D'], {'B': True, 'L': True}, i, 0))

#%%
def plot(probability, samples, alpha=1, ylim=(0.8, 1)):
    #plt.figure()
    df = pd.DataFrame({'sample_size': samples, 'probability': probability})
    plot = sns.scatterplot(x="sample_size", y="probability", ci="sd", data=df, alpha=alpha)
    #plot.set_title(str(title));
    plot.axes.set_ylim(ylim[0], ylim[1])
    return plot

plot(rejection, sample_size, 1)
plot(gibbs, sample_size, 1)
plot1 = plot(mlw, sample_size, 1)
plot1.set_title("Sample size vs estimated probability")
plot1.axes.set_xlim(100, 10000)
xticks = [100]
xticks.extend(range(1000, 10001, 1000))
plot1.axes.set_xticks(xticks);

plt.legend(labels=['rejection', 'gibbs', 'lw']);
txt="Figure 2.1. Sample size vs estimated probability for P( D | B, L )"
plt.figtext(0.5, -0.05, txt, wrap=True, ha='center', va='bottom', fontsize=10);

#%% md
In figure 2.1, we see that rejection sampling has the greatest variability, especially for very low sample sizes (< 500), where the rejection sampling probability have several points outside the graph bounds. Gibbs seems more accurate (here used without burn-in period) and likelihood-weighted sampling performs with the highest accuracy.

#%%
gibbs0 = []
gibbs1000 = []
gibbs5000 = []
for i in sample_size:
    gibbs0.append(   gibbs_sample(['D'], {'B': True, 'L': True}, i, 0))
    gibbs1000.append(gibbs_sample(['D'], {'B': True, 'L': True}, i, 1000))
    gibbs5000.append(gibbs_sample(['D'], {'B': True, 'L': True}, i, 5000))

#%%
plot(gibbs0, sample_size, 1)
plot(gibbs1000, sample_size, 1)
plot2 = plot(gibbs5000, sample_size, 1)
plot2.set_title("Comparison of different burn in periods");
xticks = [100]
xticks.extend(range(1000, 10001, 1000))
plot2.axes.set_xticks(xticks);

plt.legend(labels=['Burn in: 0', 'Burn in: 1000', 'Burn in: 5000']);
txt = "Figure 2.2. Comparison of different burn in periods for P( D | B, L )"
plt.figtext(0.5, -0.05, txt, wrap=True, ha='center', va='bottom', fontsize=10);


#%% md
Oddly enough, in figure 2.2., we see no improvement in accuracy for the gibbs method when using different burn-in periods.

#%% md
# # 3. A different query
Choose your own query (i.e. pick a conditional probability over a suitable subset of variables and estimate using the sampling methods) of this Bayes net such that the convergence and effectiveness of rejection sampling is noticeable worse than for the other two algorithms. Report which query you chose and plot the probability as a function of the number of samples used. Why is it that rejection sampling is so much worse for this example?

#%%
sample_size2 = range(1000, 100000, 1000)
rejection2 = []
mlw2 = []
gibbs2 = []
for i in sample_size2:
    rejection2.append(rejection_sample(['D'], {'V': True, 'S': True}, i))
    mlw2.append(likelihood_sample(     ['D'], {'V': True, 'S': True}, i))
    gibbs2.append(gibbs_sample(        ['D'], {'V': True, 'S': True}, i, 0))

#%%
plot(rejection2, sample_size2, 1, ylim=(0.7, 1))
plot(gibbs2, sample_size2, 0.5, ylim=(0.7, 1))
plot3 = plot(mlw2, sample_size2, 0.5, ylim=(0.7, 1))
plot3.set_title("Sample size vs estimated probability");
plt.legend(labels=['rejection', 'gibbs', 'lw']);
txt = "Figure 3.1. Sample size vs estimated probability for P( D | V, S )"
plt.figtext(0.5, -0.05, txt, wrap=True, ha='center', va='bottom', fontsize=10);



# Rejection sampling may perform worse on children with many parents, because that method doesn't take evidence into account as strongly as mlw and Gibbs.

#%% md
From figure 3.1 we see that rejection sampling seems to perform much worse when the condition nodes are far from the node whose probability we are estimating, such as for P( D | V, S ). This might be because conditioning on far away ancestors gives more room for different possibilities, whereas conditioning on parents will have the prediction correspond more closely to the node's probability table.
