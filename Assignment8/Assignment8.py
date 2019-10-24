#%% md
# Assignemnt 8
# ## Students:
- Davíð Freyr Björnsson
- Eric Guldbrand


# ### 1) The branching factor d of a directed graph is the maximum number of children (outer degree) of a node in the graph. Suppose that the shortest path between the initial state and a goal is of length r.
# ### a) What is the maximum number of BFS iterations required to reach the solution in terms of d and r?
__Answer:__ In the worst case for each depth level in the tree we check all the d children for each node. The tree depth in that case is r. Therefore we need $d \cdot r$ iterations.

# ### b) Suppose that storing each node requires one unit of memory. Hence, storing a path with k nodes requires k units of memory. What is the maximum amount of memory required for BFS in terms of d and r?
__Answer:__ If we go through the tree in the manner described in a), it would require $d^r$ units of memory in order to store the maximum queue length, that is, the length of the last row in the graph.

#%% md
# ### 2) Take the following graph where 0 and 4 are respectively the initial and the goal states. The other nodes are to be labeled by 1,2 and 3.
![image](q2.png)
# ### Suppose that in case of a tie, the DFS method takes the path with the smallest label of the last node. Show that there exists a labeling of these three nodes, where DFS will never reach to the goal!
Label 1, 2, 3 counter-clockwise. In fact, any labeling will cause an issue since 1, 2 and 3 are all less than 4.
# ### What can be added to DFS to avoid this situation?
Do iterative deepening DFS, where we first do a DFS to depth 1. Then depth 2, etc.

Mark each visited node and never go to a node that is marked as already visited.

#%% md
# ### 3) This question investigates using graph searching to design video presentations. Suppose there exists a database of video segments, together with their length in seconds and the topics covered, set up as follows:

#%% md
# ### a)
![image](3a2.jpg)

#%% md
# ### b)
