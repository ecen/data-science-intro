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
$$P(X = true, V = true) = \sum_{S,T,L,C \in \{true, false\}} P(X = true, V = true, S, T, L)$$

#%%



#%% md

# ### c. $P(C | V^c , S) = P(C = true | V = false, S = true)$

#%%
import numpy as np

class Node:
    # parents is a list of nodes
    # table is a list of probabilities, the truth table:
    #   no parents: 1 value
    #   1 parent: true, false
    #   2 parents: true true, true false, false true, false false
    def __init__(self, name="", parents=[], table=[]):
        self.name = name
        self.parents = parents
        self.table = table

    def sample(self, truths=[], lies=[]):
        for node in truths:
            if self is node:
                return True

        for node in lies:
            if self is node:
                return False

        r = np.random.uniform(size = 1)
        if len(self.parents) == 0:
            return r < self.table[0]
        elif len(self.parents) == 1:
            p1 = self.parents[0].sample(truths, lies)
            if p1:
                return r < self.table[0]
            else:
                return r < self.table[1]
        else:
            p1 = self.parents[0].sample(truths, lies)
            p2 = self.parents[1].sample(truths, lies)
            if (p1 and p2):
                return r < self.table[0]
            elif (p1 and not p2):
                return r < self.table[1]
            elif (not p1 and p2):
                return r < self.table[2]
            else:
                return r < self.table[3]

    def toString(self):
        s = ""
        for p in self.parents:
            s += p.toString() + "\n"
        s += str(self.table)
        return s

v = Node("v", table=[0.01])
t = Node("t", parents=[v], table=[0.05, 0.01])
s = Node("s", table=[0.5])
l = Node("l", parents=[s], table=[0.1, 0.01])
b = Node("b", parents=[s], table=[0.6, 0.3])
c = Node("c", parents=[t, l], table=[1, 1, 1, 0])
x = Node("x", parents=[c], table=[0.98, 0.05])
d = Node("d", parents=[c,b], table=[0.9,0.7,0.8,0.9])

def calcProb(node, truths=[], lies=[], n=100000):
    np.random.seed(42)
    t = 0
    for i in range(0, n):
        if node.sample(truths, lies):
            t += 1
    return t / n

p_d_bc = calcProb(d, [b, c])
p_x_v = calcProb(x, [v])
p_c_vcs = calcProb(c, [s], [v])

#%%
print("P(D|B,C): ", p_d_bc)
print("P(X|V): ", p_x_v)
print("P(C|V-,S): ", p_c_vcs)
