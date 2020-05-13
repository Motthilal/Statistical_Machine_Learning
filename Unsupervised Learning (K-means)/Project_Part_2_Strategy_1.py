#!/usr/bin/env python
# coding: utf-8

# In[113]:


import scipy.io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import random
from random import randrange
import operator


# In[114]:


def Euclidean_distance(pt1, pt2):
    '''
    Find the Euclidean distance between two points
    '''
    sum_ = 0

    for i in range(len(pt1)):
            sum_ += (pt1[i] - pt2[i])**2

    d = (sum_)**0.5
    return d


# In[115]:


def Initialization_Strategy_1(k,data,centroids_set):
    '''
    Calculate the initial centroids according to Strategy 1
    '''
    for i in range(k):
        centroids_set[i] = data[randrange(len(data))]
    return centroids_set


# In[116]:


def K_means(k,data):
    '''
    Run K means on the given data
    '''

    centroids = {}

    #Initialization
    centroids = Initialization_Strategy_1(k,data,centroids)
    

    for i in range(500):

        clusters = {}
        for i in range(k):
            clusters[i] = []

        #Find Euclidean Distance
        for points in data:
            distances = [Euclidean_distance(points,centroids[centroid]) for centroid in centroids]
            min_dist = distances.index(min(distances))
            clusters[min_dist].append(points)

        prev = dict(centroids)

        #Avg over centroids
        for i in clusters:
            centroids[i] = np.average(clusters[i], axis = 0)

        converge = True

        #Check convergence
        for centroid in centroids:

            org = prev[centroid]
            curr = centroids[centroid]

            objective = np.sum((curr - org)/org * 100.0)

            if objective > 0.0001:
                converge = False

        if converge:
            break

    return centroids,clusters,objective


# In[122]:


def Objective_Plot(Objective_fun):
    '''
    Plot Objective function vs K
    '''
    K_x=range(2,11,1)
    plt.plot(K_x,Objective_fun)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Value of Objective Function')
    plt.title('Objective function vs No. of Clusters (k)')
    plt.show()


# In[118]:


def Visualize_Clusters(centroids,clusters,i):
    '''
    Function to visualize clusters
    '''

    x = np.arange(10)
    ys = [i+x+(i*x)**2 for i in range(10)]

    colors = cm.rainbow(np.linspace(0, 1, len(ys)))

    for centroid in centroids:
        plt.scatter(centroids[centroid][0], centroids[centroid][1], s = 200, marker = "X", c='black')

    for i in clusters:
        color = colors[i]
        for points in clusters[i]:
            plt.scatter(points[0], points[1],color=color,s = 30)

    plt.title("Plot for k = "+str(i))
    plt.show()


# In[119]:


def main():
    raw_data = scipy.io.loadmat("AllSamples.mat")
    data = raw_data["AllSamples"]
    Obj_fun=[]
    for i in range(2,11):

        # Running K means
        centroids,clusters,obj = K_means(i,data)

        #Visualizing Clusters
        Visualize_Clusters(centroids,clusters,i)

        obj=0
        for k in range(i):
            obj+=np.sum((clusters[k]-centroids[k])**2)
        Obj_fun.append(obj)

    #Plotting Objective function
    Objective_Plot(Obj_fun)


# In[132]:


if __name__ == "__main__":
    main()

