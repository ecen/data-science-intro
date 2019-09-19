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

df200 = pd.read_csv('./Assignment3/data_500.csv')
dfall = pd.read_csv('./Assignment3/data_all.csv')

def plot180(df, alpha=1):
    plt.figure()
    plot = sns.scatterplot(x='phi', y='psi', data=df, alpha=alpha)
    plot.axes.set_xlim(-180, 180)
    plot.axes.set_ylim(-180, 180)
    plot.axes.set_xticks(range(-180, 180, 45));
    plot.axes.set_yticks(range(-180, 180, 45));
plot180(dfall, 0.01);
plot180(dfall, 0.05);
plot180(dfall, 0.1);
plot180(dfall, 1);

#%% md
2. Use the K-means clustering method to cluster the phi and psi angle combinations in the data file.
a. Select a suitable distance metric for this task. If this is different from the Euclidean distance function, explain how it differs. For a higher grade, motivate the choice of distance metric.

Euclidian distance was chosen as the distance metric. Since taking means of degrees that can range from -180 to 180 can give nonsensical results, we mapped the angles to Euclidian space using the function $$f(v) := [cos(v), sin(v)]$$, where v is an angle in radians. As such each point in the 2-dimensional toroidal input space becomes a 4-dimensional vector $$[cos(phi), sin(phi), cos(psi), sin(psi)]$$ in Euclidian space. With that there is increased risk of our analysis being adversly affected by the curse of dimensionality. The main risk being that high dimensional datasets are likely to be very sparse, which can lead to overfitting. The number of observations needed to combat this, grows exponentially with the number of dimensions. So for a 4-dimensional dataset we need approximately 55 observations. Thankfully we therefore have more than enough observations.

#%%
# Map the angles to Euclidian space
def toEuclidian4(df):
    df['phi_x'] = df['phi'].apply(lambda x : math.cos(math.radians(x)))
    df['phi_y'] = df['phi'].apply(lambda x : math.sin(math.radians(x)))
    df['psi_x'] = df['psi'].apply(lambda x : math.cos(math.radians(x)))
    df['psi_y'] = df['psi'].apply(lambda x : math.sin(math.radians(x)))
    return df;
dfall = toEuclidian4(dfall)

print(dfall.head())

def plot360(df):
    plot = sns.scatterplot(x='phi', y='psi', data=df);
    epsilon = 10
    plot.axes.set_xlim(-180-epsilon, 180+epsilon);
    plot.axes.set_ylim(-180-epsilon, 180+epsilon);
    plot.axes.set_xticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
    plot.axes.set_yticks(range(-180-epsilon, 180+epsilon, 4*epsilon));
plot360(dfall);

#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%
# K-means clustering
from sklearn.cluster import KMeans

# Plots a dataframe using its 'cluster' column as hue
def plotClusters(df):
    plot = sns.scatterplot(x='phi', y='psi', data=df, hue=df['cluster']);
    plot.axes.set_xlim(-180, 180);
    plot.axes.set_ylim(-180, 180);
    plot.axes.set_xticks(range(-180, 180, 45));
    plot.axes.set_yticks(range(-180, 180, 45));
    return plot

def plotKmeans(k, df, addToTitle=""):
    plt.figure()
    kmeans = KMeans(n_clusters=k, random_state=0).fit(df[['phi_x', 'phi_y', 'psi_x', 'psi_y']])
    df['cluster'] = kmeans.labels_

    plot = plotClusters(df)
    plot.set_title("Clustering, k = " + str(k) + addToTitle);
    return kmeans

kmeans = {}
for i in range(1, 8):
    kmeans[i] = plotKmeans(i, dfall)

#%% md
Trying several different values of k for the we find that the small dataset (n=200) is neatly divided into k=3 clusters. However, k=4 also works quite well, splitting of one part of the top-right cluster that is somewhat stretched out from the rest of the grouping.

#%% Elbow test
def elbowTest(kmeans):
    sse = {} # Sum of Squared Errors
    for i in range(1, 8):
        sse[i] = kmeans[i].inertia_
    return sse
sse = elbowTest(kmeans)
plt.plot(list(sse.keys()), list(sse.values()));
plt.xlabel('K (number of clusters)')
plt.ylabel('Absolute error')

#%% md
Similarily to visual inspection, the elbow test also points to a k=3 or k=4 as the best options. We decide to use k=3 due to how neatly the clusters are visually grouped into 3.

#%% md
__c.__ Validate the clusters that are found with the chosen value of K.

#%%
# Test stability on subsets
def randomSubset(df, nPercent):
    n = round((nPercent/100.0)*len(df))
    np.random.seed(n)
    indices_to_drop = np.random.choice(df.index, n, replace=False)
    return df.drop(indices_to_drop)

# Plot the results of KMeans clustering, when 0, 10, 20, 30%
# of observations are dropped
percentDropped = [0, 10, 20, 30]
for percent in percentDropped:
    subset = randomSubset(dfall, percent)
    plotKmeans(3, subset, ", " + str(percent) + " percent of observations dropped")

#%% md
The scatterplots show that the K-means clustering is stable over different subsets of the original dataframe.

#%% md
3. Use the DBSCAN method to cluster the phi and psi angle combinations in the data file.
__a.__ Motivate:

__i.__ the choice of the minimum number of samples in the neighbourhood
for a point to be considered as a core point, and

We choose 0.5\% of the total number of samples to be the min_samples value. A higher value causes DBSCAN to not recognize the rightmost cluster at all since it is much less dense than the two main clusters. However, recognizing all three as seperate clusters seems important since while much less dense, the rightmost cluster is clearly distinct from the first two from visual inspection. This could indicate an energy-level that is not as low as the other two, but still stable.

__ii.__ the choice of the maximum distance between two samples belonging
to the same neighbourhood (“eps” or “epsilon”).

Due to how the data has been transformed into 4D euclidian space, epsilon will correlate with the number of degrees in one of the 2 toroid dimensions. As such, eps=2 will be all points with +-180 degrees, that is, all points in the data. eps=1 is +-90 degrees, and so on.

We choose eps=0.25 => +- 22.5 degrees, that is, points within a cone of 45 degrees. This seems like a reasonably large value and allows us to, along with our min_samples value, recognize similar clusters as with 3-means, even though the rightmost cluster is much smaller and less dense than the two others.



#%%
from sklearn.cluster import DBSCAN

dfall = toEuclidian4(dfall);
def plotDbscan(df):
    clustering = DBSCAN(eps=0.25, min_samples=len(df)*0.005).fit(df[['phi_x', 'phi_y', 'psi_x', 'psi_y']])
    df['cluster'] = clustering.labels_
    plotClusters(df)
    return df
plotDbscan(dfall);
#df200 = plotDbscan(df200);




#%% md
__b.__ Highlight the clusters found using DBSCAN and any outliers in a scatter plot.
How many outliers are found? Plot a histogram to show which amino acid
residue types are most frequently outliers.

#%% md
c. Compare the clusters found by DBSCAN with those found using K-means.

#%% md
d. Discuss whether the clusters found using DBSCAN are robust to small changes
in the minimum number of samples in the neighbourhood for a point to be considered as a core point, and/or the choice of the maximum distance between two samples belonging to the same neighbourhood (“eps” or “epsilon”).




#%% md
# ## Bibliography
Géron, A. (2017). Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".

Skiena, S. S. (2017). The data science design manual. Springer.
