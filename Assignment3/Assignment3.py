#%% md
# # Assignment 3
* Students:
    * Davíð Freyr Björnsson
    * Eric Guldbrand
* Time spent per person:

#%% md
1. Draw a scatter plot that shows the phi and psi combinations in the data file.

#%%
import pandas as pd
import matplotlib as pl
import seaborn as sns

df = pd.read_csv('./Assignment3/data_200.csv')
df.head()
sns.set()

plt = sns.scatterplot(x='phi', y='psi', data=df)
plt.axes.set_xlim(-180, 180)
plt.axes.set_ylim(-180, 180)

#%% md
2. Use the K-means clustering method to cluster the phi and psi angle combinations in the data file.
a. Select a suitable distance metric for this task. If this is different from the Euclidean distance function, explain how it differs. For a higher grade, motivate the choice of distance metric.



#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%


#%% md
c. Validate the clusters that are found with the chosen value of K.

#%%
