#!/usr/bin/env python
# coding: utf-8

# ### ## K-Means Clustering 
# 1. Create the dataset and plot 
# 2. Define number of clusters K and initialized with empty list ( later we will insert points index in this list)
#     - Clusters is a list of list
#     - Inner list consist of the  points or samples which are clossest to the centroids 
# 
# 3. define the initial centroids c_1, c_2 with random value ( 2 random sample/point) 
# 
# 3. Repeat steps 4 and 5 until convergence or until the end of a fixed number of iterations
# 4. for each data point x_i: 
# 
# 
#       - a. Find the distance of an sample with c_1 and c_2
#       - b. if the the sample is close to the c_1
#           - save the value in clusters list of index 0 ( Remeber clusters is a list of list as we have two cluster the clusters             list length is 2 (index o and 1)
#           
#       - c. if the the sample is close to the c_2
#           - save the value in clusters list of index 1 ( Remeber clusters is a list of list as we have two cluster the clusters             list length is 2 (index o and 1)  
#           
#       - d. using a,b,c sub_steps we are separating the data into two part
#       
#       - get the centroid index find the nearest centroid(c_1, c_2 .. c_k) 
#        - assign the point to that cluster 
# 5. for each cluster j = 1..k
#        - new centroid = mean of all points assigned to that cluster
# 6. End

# ### Create the dataset and plot

# In[10]:


import matplotlib.pyplot as plt
import numpy as np

np.random.seed(42)

from sklearn.datasets import make_blobs

#creates dataset 
X, y = make_blobs(
    centers=3, n_samples=20, n_features=2, shuffle=True, random_state=40
)
#plot the dataset
plt.scatter(X[:,0], X[:,1])
plt.show()


# ### 2. Define number of clusters K and initialized with empty lists ( later we will insert points index in this list)
# - Cluster is a list of list
#  -intial clusters= [[]] 
#        
# - Inner list consist of the  points or samples which are clossest to the centroids 
#      - later clusters=[[all points close to c1], all points close to c2]

# In[12]:


# first we need to chose the number of cluster, lets assume it is 2
number_of_cluster=3

#Then we need to create a list of list with length of 2 which contains all the points inside the cluster
# initially it is containing 2 empty list keeping in mind that we have two clusters, later we will append the datapoints which are closest to each clusters centroids
clusters = [[] for _ in range(number_of_cluster)]
print(clusters)


# ## 3. define the initial centroids c_1, c_2 with random value ( 2 random sample/point) 

# In[22]:


# Randomly initialize 2 centroids
# Why it is 2 because we have two clusters (print the length of the clusters)
# initialize centroid
#We can use np.random.choice generator which takes length of smaple and number of cluster
n_samples,number_of_features=X.shape
random_centroid_idxs = np.random.choice(n_samples, number_of_cluster, replace=False)
centroids= [X[idx] for idx in random_centroid_idxs]

print(centroids)

#plot the dataset
#plot the centroids
def plot():
    fig, ax = plt.subplots(figsize=(12, 8))

    #for  index in range(len(X)):
     #   point = X[index].T
      #  ax.scatter(*point)

    plt.scatter(X[:,0], X[:,1],color="red")
    for point in centroids:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
plot()


# ## 4. for sample in  X:

# 
# > - 1. Find the distance of an sample with c_1 and c_2
# 
# > - 2. Find the centroid_id closest to the point ( **find is it c_1 or is it c_2?**)
# 
# > - 3. if the the sample is close to the c_1
#     - save the value in clusters list of index 0 ( Remeber clusters is a list of list as we have two cluster the clusters             list length is 2 (index o and 1)
#           
# > - 4. if the the sample is close to the c_2
#      - save the value in clusters list of index 1 ( Remeber clusters is a list of list as we have two cluster the clusters             list length is 2 (index o and 1)  
#           
# - e. using a,b,c sub_steps we are separating the data into two part
# 
# 
# > - 5. Find the new_centroids
#     - c1_mean= calculate the mean of the  points inside cluster[0]
#     - new_centroid_1=c1_mean
#     - c2_mean= calculate the mean of the  points inside cluster[1]
#     - new_centroid_2=c2_mean 
#     -centroids=[new_centroid_1,new_centroid_2]

# In[60]:


def find_distance(sample, centroid):
    return np.sqrt(np.sum((sample - centroid) ** 2))

def closest_centroid(sample,centroids):
    distances=[find_distance(sample,centroid) for centroid in centroids]
    cl_centroid=np.argmin(distances)
    return cl_centroid

def sample_mean_in_cluster(cluster):
    
    cluster_mean = np.mean(X[cluster], axis=0)
    
    return cluster_mean
def _mean(cluster):
    cluster_mean = np.mean(cluster)
    
    return cluster_mean
    
def update_cluster_and_centroid():
    clusters = [[] for _ in range(number_of_cluster)]
    print(clusters)
    n_samples,number_of_features=X.shape
    random_centroid_idxs = np.random.choice(n_samples, number_of_cluster, replace=False)
    centroids= [X[idx] for idx in random_centroid_idxs]
    for idx,sample in enumerate(X):
        cl_centroid=closest_centroid(sample,centroids)
        if cl_centroid==0:
            clusters[0].append(idx)
        if cl_centroid==1:
            clusters[1].append(idx)
        if cl_centroid==2:
            clusters[2].append(idx)
    
    centroids=[sample_mean_in_cluster(clusters[0]),sample_mean_in_cluster(clusters[1]),sample_mean_in_cluster(clusters[2])]
    return clusters, centroids


def _is_converged(centroids_old, centroids):
    # distances between each old and new centroids, fol all centroids
    distances = [
        euclidean_distance(centroids_old[i], centroids[i]) for i in range(number_of_cluster)
    ]
    return sum(distances) == 0

def t_plot(clusters,centroids):
    fig, ax = plt.subplots(figsize=(12, 8))

    for i, index in enumerate(clusters):
        point = X[index].T
        ax.scatter(*point)

    #plt.scatter(X[:,0], X[:,1],color="red")
    for point in centroids:
        ax.scatter(*point, marker="x", color="black", linewidth=2)

    plt.show()
    
for it in range(100):
    old_centroids=centroids

    clusters, centroids=update_cluster_and_centroid()
    #t_plot(clusters,centroids)
    print(sample_mean_in_cluster(clusters[0]))
    if (_is_converged(old_centroids,centroids)):
        print("Converged")
        break
        
    
    
    
    

    


#print(clusters)  
#print(centroids)  


# In[ ]:




