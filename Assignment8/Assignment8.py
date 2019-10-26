#%% md
# Assignemnt 8
# ## Students:
- Davíð Freyr Björnsson
- Eric Guldbrand


# ### 1) The branching factor d of a directed graph is the maximum number of children (outer degree) of a node in the graph. Suppose that the shortest path between the initial state and a goal is of length r.
# ### a) What is the maximum number of BFS iterations required to reach the solution in terms of d and r?
__Answer:__ In the worst case we need to check all nodes in the tree. Therefore we need $\sum_{i=0}^{r} d^i$ iterations.

# ### b) Suppose that storing each node requires one unit of memory. Hence, storing a path with k nodes requires k units of memory. What is the maximum amount of memory required for BFS in terms of d and r?
__Answer:__ If we go through the tree in the manner described in a), it would require $d^r$ units of memory in order to store the maximum queue length, that is, the length of the last (and largest) row in the graph.

#%% md
# ### 2) Take the following graph where 0 and 4 are respectively the initial and the goal states. The other nodes are to be labeled by 1,2 and 3.
![image](q2.png)
# ### Suppose that in case of a tie, the DFS method takes the path with the smallest label of the last node. Show that there exists a labeling of these three nodes, where DFS will never reach to the goal!
Label 1, 2, 3 counter-clockwise. In fact, any labeling will cause an issue since 1, 2 and 3 are all less than 4.
# ### What can be added to DFS to avoid this situation?
Mark each visited node and never go to a node that is marked as already visited.

#%% md
# ### 3) This question investigates using graph searching to design video presentations. Suppose there exists a database of video segments, together with their length in seconds and the topics covered, set up as follows:

#%% md
# ### a)
![image](3a2.jpg)

#%% md
# ### b)
One heuristic is h(n) as the total cost from the start node to n. The frontier is sorted with lowest h(n) first. This heuristic will be less than the cost for the final path up until h(goal node) where they will be equal.

#%% md
# ## 4)
Each grid square is labeled using a row index (letter) and column index (number). In case of tie, algorithm chooses the node with the lowest lexicographical value (eg. A1 < A2 < B1)
![](labeled-grid.png)

The explored paths are the following:
```
[F4, E4]
[F4, E4, E3]
[F4, E4, E5]
[F4, F5]
[F4, G4]
```

#%% md
![BFS](BFS)
Best-first performs best in this case since the wall structure doesn't have any significant dead-ends, which seems likely to punish a greedy algorithm more. It is fine to just follow the heuristic.

![A star](A_star)
A* performs a little worse since it is more indecisive on which path to follow, and sorting based on distance AND heuristic isn't beneficial enough in this quite simple case.

![Best first search](Best-first-search)
BFS performs the worst since the wall structure means that it will spend alot of time search in the opposite direction of the goal. Of course, BFS doesn't have any heuristic to help it know where it might be good to search.
