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
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

df = pd.read_csv('./Assignment3/data_200.csv')
print(df.describe())
sns.set()

def plot180():
    plot = sns.scatterplot(x='phi', y='psi', data=df)
    plot.axes.set_xlim(-180, 180)
    plot.axes.set_ylim(-180, 180)
    plot.axes.set_xticks(range(-180, 180, 45));
    plot.axes.set_yticks(range(-180, 180, 45));
plot180()

#%% md
2. Use the K-means clustering method to cluster the phi and psi angle combinations in the data file.
a. Select a suitable distance metric for this task. If this is different from the Euclidean distance function, explain how it differs. For a higher grade, motivate the choice of distance metric.

Euclidian distance was chosen as the distance metric. We mapped the angles to Euclidian space using the function $$f(v) := [cos(v), sin(v)]$$, where v is an angle in radians. As such each point in the 2-dimensional toroidal input space becomes a 4-dimensional vector $$[cos(phi), sin(phi), cos(psi), sin(psi)]$$ in Euclidian space.

#%%
# Map the angles to Euclidian space
df['phi_x'] = df['phi'].apply(lambda x : math.cos(math.radians(x)))
df['phi_y'] = df['phi'].apply(lambda x : math.sin(math.radians(x)))
df['psi_x'] = df['psi'].apply(lambda x : math.cos(math.radians(x)))
df['psi_y'] = df['psi'].apply(lambda x : math.sin(math.radians(x)))

print(df.head())
#print(df.describe())

def plot360():
    plot = sns.scatterplot(x='phi', y='psi', data=df);
    epsilon = 10
    plot.axes.set_xlim(-180-epsilon, 180+epsilon);
    plot.axes.set_ylim(-180-epsilon, 180+epsilon);
    plot.axes.set_xticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
    plot.axes.set_yticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
plot360();

#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%
# K-means clustering
from sklearn.cluster import KMeans
def plot360Kmeans(k):
    plt.figure()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df[['phi_x', 'phi_y', 'psi_x', 'psi_y']])
    df['cluster'] = kmeans.labels_

    plot = sns.scatterplot(x='phi', y='psi', data=df, hue=df['cluster']);
    plot.axes.set_xlim(-180, 180);
    plot.axes.set_ylim(-180, 180);
    plot.axes.set_xticks(range(-180, 180, 45));
    plot.axes.set_yticks(range(-180, 180, 45));
    return kmeans

kmeans = {}
for i in range(1, 10):
    kmeans[i] = plot360Kmeans(i)

#%% md
c. Validate the clusters that are found with the chosen value of K.
#%%

sse = {} # Sum of Squared Errors
for i in range(1, 10):
    sse[i] = kmeans[i].inertia_

plt.plot(list(sse.keys()), list(sse.values()));
