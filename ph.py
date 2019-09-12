# -*- coding: utf-8 -*-
"""
Created on Wed Sep 11 23:56:16 2019

@author: jaehooncha
"""
from ripser import ripser
from persim import plot_diagrams
import tadasets

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from scipy import sparse
from matplotlib import collections

def plot_balls(data, r):
    patches = [plt.Circle(center, r) for center in data]
    fig, ax = plt.subplots()
    coll = collections.PatchCollection(patches, edgecolors='black', facecolors='r', alpha=0.5)
    ax.add_collection(coll)
    ax.set_aspect('equal', adjustable='datalim')
    ax.plot()   #Causes an autoscale update.


def getGreedyPerm(D):
    """
    A Naive O(N^2) algorithm to do furthest points sampling

    Parameters
    ----------
    D : ndarray (N, N)
        An NxN distance matrix for points

    Return
    ------
    lamdas: list
        Insertion radii of all points
    """

    N = D.shape[0]
    #By default, takes the first point in the permutation to be the
    #first point in the point cloud, but could be random
    perm = np.zeros(N, dtype=np.int64)
    lambdas = np.zeros(N)
    ds = D[0, :]
    for i in range(1, N):
        idx = np.argmax(ds)
        perm[i] = idx
        lambdas[i] = ds[idx]
        ds = np.minimum(ds, D[idx, :])
    return lambdas[perm]

def getApproxSparseDM(lambdas, eps, D):
    """
    Purpose: To return the sparse edge list with the warped distances, sorted by weight

    Parameters
    ----------
    lambdas: list
        insertion radii for points
    eps: float
        epsilon approximation constant
    D: ndarray
        NxN distance matrix, okay to modify because last time it's used

    Return
    ------
    DSparse: scipy.sparse
        A sparse NxN matrix with the reweighted edges
    """
    N = D.shape[0]
    E0 = (1+eps)/eps
    E1 = (1+eps)**2/eps

    # Create initial sparse list candidates (Lemma 6)
    # Search neighborhoods
    nBounds = ((eps**2+3*eps+2)/eps)*lambdas

    # Set all distances outside of search neighborhood to infinity
    D[D > nBounds[:, None]] = np.inf
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    idx = I < J
    I = I[(D < np.inf)*(idx == 1)]
    J = J[(D < np.inf)*(idx == 1)]
    D = D[(D < np.inf)*(idx == 1)]

    #Prune sparse list and update warped edge lengths (Algorithm 3 pg. 14)
    minlam = np.minimum(lambdas[I], lambdas[J])
    maxlam = np.maximum(lambdas[I], lambdas[J])

    # Rule out edges between vertices whose balls stop growing before they touch
    # or where one of them would have been deleted.  M stores which of these
    # happens first
    M = np.minimum((E0 + E1)*minlam, E0*(minlam + maxlam))

    t = np.arange(len(I))
    t = t[D <= M]
    (I, J, D) = (I[t], J[t], D[t])
    minlam = minlam[t]
    maxlam = maxlam[t]

    # Now figure out the metric of the edges that are actually added
    t = np.ones(len(I))

    # If cones haven't turned into cylinders, metric is unchanged
    t[D <= 2*minlam*E0] = 0

    # Otherwise, if they meet before the M condition above, the metric is warped
    D[t == 1] = 2.0*(D[t == 1] - minlam[t == 1]*E0) # Multiply by 2 convention
    return sparse.coo_matrix((D, (I, J)), shape=(N, N)).tocsr()


def makeSparseDM(X, thresh):
    N = X.shape[0]
    D = pairwise_distances(X, metric='euclidean')
    [I, J] = np.meshgrid(np.arange(N), np.arange(N))
    I = I[D <= thresh]
    J = J[D <= thresh]
    V = D[D <= thresh]
    return sparse.coo_matrix((V, (I, J)), shape=(N, N)).tocsr()


data1 = tadasets.infty_sign(n=2000, noise=0.1)


data2 = np.concatenate([
    tadasets.dsphere(n=500, d=1, r=5, noise=0.5),
    tadasets.dsphere(n=100, d=1, r=1, noise=0.2)
])

data3 = np.concatenate([
    tadasets.dsphere(n=300, d=1, r=2, noise=0.5)+np.array([-2,2.3]),
    tadasets.dsphere(n=200, d=1, r=1.5, noise=0.2)+np.array([1.5,2.6]),
    tadasets.dsphere(n=100, d=1, r=1, noise=0.2),
])
 

def plot_Graph(X):
    eps = 0.1
    
    # Compute the sparse filtration
    # First compute all pairwise distances and do furthest point sampling
    D = pairwise_distances(X, metric='euclidean')
    lambdas = getGreedyPerm(D)
    
    # Now compute the sparse distance matrix
    DSparse = getApproxSparseDM(lambdas, eps, D)
    
    # Finally, compute the filtration
    resultsparse = ripser(DSparse, distance_matrix=True)
    
    lifetime = resultsparse['dgms'][1][:,1]-resultsparse['dgms'][1][:,0]
    mask_holes = np.where(lifetime> np.percentile(lifetime,99))[0]
      
    plot_diagrams(resultsparse['dgms'], show=False)
    plt.scatter(resultsparse['dgms'][1][mask_holes, 0], 
            resultsparse['dgms'][1][mask_holes, 1], c = 'r', s = 100, alpha=0.3)






