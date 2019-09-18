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

Euclidian distance was chosen as the distance metric. Prior to calculating the distance, normalization of the data was carried out, so that the phi and psi angles range from 0 to 360 degrees. A corresponding distance function returns the shortest Euclidian distance between the angles.
#%%
# Distance calculation
def distance(degree1, degree2):
    return min((degree1 - degree2) % 360, (degree2 - degree1) % 360)

# Test it
print(distance(45, 315)) # = 90
print(distance(190, 350)) # = 160
print(distance(190, 170)) # = 20
print(distance(-170, 170)) # = 20
print(distance(-170, -10)) # = 160

#%%
# Remap data to [0, 360] range for clarity.
df['phi'] = df['phi'].apply(lambda x : x % 360)
df['psi'] = df['psi'].apply(lambda x : x % 360)
#print(df.describe())

def plot360():
    plt = sns.scatterplot(x='phi', y='psi', data=df);
    plt.axes.set_xlim(0, 360);
    plt.axes.set_ylim(0, 360);
    plt.axes.set_xticks(range(0, 360, 45));
    plt.axes.set_yticks(range(0, 360, 45));
plot360();

#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%
# K-means clustering
from sklearn.cluster import KMeans
def plot360Kmeans():
    kmeans = KMeans(n_clusters=3, random_state=0).fit(df[['phi', 'psi']])
    df['cluster'] = kmeans.labels_

    plt = sns.scatterplot(x='phi', y='psi', data=df, hue=df['cluster']);
    plt.axes.set_xlim(0, 360);
    plt.axes.set_ylim(0, 360);
    plt.axes.set_xticks(range(0, 360, 45));
    plt.axes.set_yticks(range(0, 360, 45));
plot360Kmeans();

#%% md
c. Validate the clusters that are found with the chosen value of K.
