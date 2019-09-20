#%% md
# # Assignment 3
* Students:
    * Davíð Freyr Björnsson
    * Eric Guldbrand
* Time spent per person: 10 hours

#%% md
# ## 1. Draw a scatter plot that shows the phi and psi combinations in the data file.

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

df200 = pd.read_csv('./data_500.csv')
dfall = pd.read_csv('./data_all.csv')

def plot180(df, alpha=1):
    plt.figure()
    plot = sns.scatterplot(x='phi', y='psi', data=df, alpha=alpha)
    plot.set_title(str((1-alpha)*100) + "% opacity");
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

Euclidian distance was chosen as the distance metric since most relevant functions will work out of the box using it. However, since taking means of degrees that range from -180 to 180 can give nonsensical results, we mapped the angles to Euclidian space using the function $$f(v) := [cos(v), sin(v)]$$, where v is an angle in radians. As such each point in the 2-dimensional toroidal input space becomes a 4-dimensional vector $$[cos(phi), sin(phi), cos(psi), sin(psi)]$$ in Euclidian space, where using euclidian distance makes sense.

There could be an increased risk of our analysis being adversly affected by the curse of dimensionality. High dimensional datasets are likely to be very sparse, which can lead to overfitting. The number of observations needed to combat this, grows exponentially with the number of dimensions. But this is more of a problem where features are independent, but here it's not the case since the trigonometric functions are very much related to each other. Additionally, we have alot of observations (about 30.000) so our analysis shouldn't be much affected by the curse.

#%%
# Map the angles to Euclidian space
def toEuclidian4(df):
    df['phi_x'] = df['phi'].apply(lambda x : math.cos(math.radians(x)))
    df['phi_y'] = df['phi'].apply(lambda x : math.sin(math.radians(x)))
    df['psi_x'] = df['psi'].apply(lambda x : math.cos(math.radians(x)))
    df['psi_y'] = df['psi'].apply(lambda x : math.sin(math.radians(x)))
    return df;
dfall = toEuclidian4(dfall)

#%% md
b. Experiment with different values of K. Suggest an appropriate value of K for this task and motivate this choice.

#%%
# K-means clustering
from sklearn.cluster import KMeans

# Plots a dataframe using its 'cluster' column as hue
def plotClusters(df, title=""):
    plot = sns.scatterplot(x='phi', y='psi', data=df, hue=df['cluster']);
    plot.set_title(title);
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
Trying several different values of k for the we find that the small dataset (n=200) is most neatly divided into k=3 clusters.

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
Similarily to visual inspection, the elbow test points to k=3 or k=4 as the best options. We decide to use k=3 due to how neatly the clusters are visually grouped into 3.

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
# ## 3. Use the DBSCAN method to cluster the phi and psi angle combinations in the data file.
__a.__ Motivate:

__i.__ the choice of the minimum number of samples in the neighbourhood
for a point to be considered as a core point, and

We choose 0.5\% of the total number of samples to be the min_samples value. A higher value causes DBSCAN to not recognize the rightmost cluster at all since it is much less dense than the two main clusters. However, recognizing all three as seperate clusters seems important since while much less dense, the rightmost cluster is clearly distinct from the first two from visual inspection. This could indicate an energy-level that is not as low as the other two, but still stable.

__ii.__ the choice of the maximum distance between two samples belonging
to the same neighbourhood (“eps” or “epsilon”).

Due to how the data has been transformed into 4D euclidian space, epsilon will correlate with the number of degrees in one of the 2 toroid dimensions. As such, eps=2 will be all points with +-180 degrees, that is, all points in the data. eps=1 is +-90 degrees, and so on.

We choose eps=0.25 => +- 22.5 degrees, that is, points within a cone of 45 degrees. This seems like a reasonably large value and allows us to, along with our min_samples value, recognize similar clusters as with 3-means, even though the rightmost cluster is much smaller and less dense than the two others.

#%% md
__b.__ Highlight the clusters found using DBSCAN and any outliers in a scatter plot.
How many outliers are found? Plot a bar chart to show which amino acid
residue types are most frequently outliers.
#%%
# Scatterplot of clusters using DBSCAN
from sklearn.cluster import DBSCAN

dfall = toEuclidian4(dfall);
def plotDbscan(df, eps, min_samples, addToTitle=""):
    clustering = DBSCAN(eps=0.25, min_samples=min_samples).fit(df[['phi_x', 'phi_y', 'psi_x', 'psi_y']])
    df['cluster'] = clustering.labels_
    title = "DBSCAN plot, " + addToTitle
    plotClusters(df, title)
    return df
dfclustered = plotDbscan(dfall, 0.25, len(dfall)*0.005);

#%%
# Number of outliers
dfoutliers = dfclustered[dfclustered['cluster'] == -1]
print(len(dfoutliers))
print(len(dfoutliers[dfoutliers["residue name"] == "GLY"]))
plot = sns.countplot(x="residue name", data=dfoutliers);
plt.xticks(rotation=90);

#%%
# Comparison with full dataset (note: NOT the same order or colors)
# Rules out that the full dataset just has much more GLY
plt.figure()
plotpercent = sns.countplot(x="residue name", data=dfall);
plt.xticks(rotation=90);


#%% md
In total there are 1852 outliers, with 845 of those being of the GLY type. There are quite a few amino acid residues of the GLY type but not enough to explain the number of outliers, so the GLY type is definitely an anomaly in that respect.

#%% md
__c.__ Compare the clusters found by DBSCAN with those found using K-means.

The main difference is that K-means has to classify everything into one of the clusters whereas DBSCAN says that anything that isn't in a tight enough cluster, is an outlier.

DBSCAN finds three clusters, each located around similar centers as the clusters found by K-means.


#%% md
__d.__ Discuss whether the clusters found using DBSCAN are robust to small changes in the minimum number of samples in the neighbourhood for a point to be considered as a core point, and/or the choice of the maximum distance between two samples belonging to the same neighbourhood (“eps” or “epsilon”).
#%%
eps = [0.15, 0.25, 0.35]
p_points = [0.004, 0.005, 0.006]
for epsilon in eps:
    for percentage in p_points:
        plt.figure();
        titleText = "epsilon: " + str(epsilon) + ", min no of samples (% of samples): " + str(percentage*100)
        plotDbscan(dfall, epsilon, len(dfall)*percentage, addToTitle=titleText);
#%% md
The clusters found using DBSCAN are robust with respect to the minimum number of samples, since the clusters found by DBSCAN are quite dense.

We see that slightly increasing the minimum number of samples (expressed as the % of samples) from 0.4\% to 0.5\% or 0.6\% has the effect of splitting cluster 1 into clusters 0 and 1. This happens independently of the value of epsilon. The border between clusters 1 and 2 is therefore quite unstable. Cluster 2 only decreases in size when the minimum sample size is increased and epsilon is held constant, and is therefore more stable. If we vary the value for epsilon but hold the minimum number of samples constant, all clusters are quite stable, both in size and number.


#%% md
# ## 4. Stratification of amino acid type
The data file can be stratified by amino acid residue type. Investigate how the clusters found for amino acid residues of type PRO differ from the general clusters. Similarly, investigate how the clusters found for amino acid residues of type GLY differ from the general clusters. Remember that parameters might have to be adjusted from those used in previous questions.

#%%
dfpro = dfall[dfall["residue name"] == "PRO"].copy()
dfgly = dfall[dfall["residue name"] == "GLY"].copy()

plotDbscan(dfpro, 0.25, len(dfpro)*0.05, "PRO");
plt.figure();
plotDbscan(dfgly, 0.25, len(dfpro)*0.05, "GLY");

#%% md
Plotting PRO and GLY with 5\% of points used as the minimum number of samples (instead of 0.5\% as for the full dataset) shows where these residues are clustered most tightly.

PRO is located almost entirely within the two biggest clusters of the full dataset. Although its a less dense area it still follows the path, where the two main clusters in the full dataset share a border.

GLY has many areas that are not very tightly clustered, but has three main clusters, one in each cluster of the main dataset. Interestingly, its rightmost cluster located in the full dataset's smallest cluster, is spread over an area similar in size to that of the full cluster. However, GLY's clusterings in the two biggest clusters are much tighter in comparison.

#%% md
# ## Bibliography
Géron, A. (2017). Hands-on machine learning with Scikit-Learn and TensorFlow: concepts, tools, and techniques to build intelligent systems. " O'Reilly Media, Inc.".

Skiena, S. S. (2017). The data science design manual. Springer.
