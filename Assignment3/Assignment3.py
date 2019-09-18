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
import numpy as np
import math

df = pd.read_csv('./Assignment3/data_200.csv')
print(df.describe())
sns.set()

def plot180():
    plt = sns.scatterplot(x='phi', y='psi', data=df)
    plt.axes.set_xlim(-180, 180)
    plt.axes.set_ylim(-180, 180)
    plt.axes.set_xticks(range(-180, 180, 45));
    plt.axes.set_yticks(range(-180, 180, 45));
plot180()

#%% md
2. Use the K-means clustering method to cluster the phi and psi angle combinations in the data file.
a. Select a suitable distance metric for this task. If this is different from the Euclidean distance function, explain how it differs. For a higher grade, motivate the choice of distance metric.

Euclidian distance was chosen as the distance metric. We mapped the angles to Euclidian space using the function $$f(v) := [cos(v), sin(v)]$$, where v is an angle in radians.

#%%
# Map the angles to Euclidian space
df['phi_x'] = df['phi'].apply(lambda x : math.cos(math.radians(x)))
df['phi_y'] = df['phi'].apply(lambda x : math.sin(math.radians(x)))
df['psi_x'] = df['psi'].apply(lambda x : math.cos(math.radians(x)))
df['psi_y'] = df['psi'].apply(lambda x : math.sin(math.radians(x)))

print(df.head())
#print(df.describe())

def plot360():
    plt = sns.scatterplot(x='phi', y='psi', data=df);
    epsilon = 10
    plt.axes.set_xlim(-180-epsilon, 180+epsilon);
    plt.axes.set_ylim(-180-epsilon, 180+epsilon);
    plt.axes.set_xticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
    plt.axes.set_yticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
plot360();

#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%
# K-means clustering
from sklearn.cluster import KMeans
def plot360Kmeans():
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['phi_x', 'phi_y', 'psi_x', 'psi_y']])
    df['cluster'] = kmeans.labels_

    plt = sns.scatterplot(x='phi', y='psi', data=df, hue=df['cluster']);
    plt.axes.set_xlim(-180, 180);
    plt.axes.set_ylim(-180, 180);
    plt.axes.set_xticks(range(-180, 180, 45));
    plt.axes.set_yticks(range(-180, 180, 45));
plot360Kmeans()

#%% md
c. Validate the clusters that are found with the chosen value of K.
