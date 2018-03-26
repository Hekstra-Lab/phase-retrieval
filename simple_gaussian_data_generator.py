#!usr/bin/python

""" Create 10 sets of 1000 images each containing up to 50 randomly placed,
    nonoverlapping gaussians. Each gaussian has covariance matrix equal to 
    10*I. Also save a file containing the fourier magnitudes of the corresponding
    images. 
""" 


import numpy as np
from scipy.stats import multivariate_normal
import os

def overlap(x1,x2,r):
    if np.any(np.linalg.norm(x1-x2,axis=1) < r):
        return True
    else:
        return False

def normalize_image(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def simple_gaussians(N, r, size=128):
    """ Create an image containing N non-overlapping gaussians each with covariance matrices proportional
        to r*I. Default sixe is 100x100"""
    uc = np.zeros((size,size))
    X,Y = np.meshgrid(np.arange(size),np.arange(size))
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    means = np.zeros((N,2))
    for i in range(N):
        while True:
            test = np.random.randint(int(r/2),int(size-r/2),size=2) #Dont let the gaussian go over the edges
            if not overlap(means,test,r):
                break
        means[i]=test
        mvn = multivariate_normal(mean=test,cov = r)
        uc = uc + mvn.pdf(pos)
    #print means
    return uc

os.makedirs("gaussian_data", exist_ok=True)

for N in range(10):
    mags = np.zeros((1000,128,128))
    ims = np.zeros((1000,128,128))
    for i in range(1000):
        n = np.random.randint(1,51)
        ims[i] = normalize_image(simple_gaussians(n,10))
        mags[i] = np.abs(np.fft.fft2(ims[i]))
    np.save("gaussian_data/mags{}.npy".format(N),mags)
    np.save("gaussian_data/imgs{}.npy".format(N),ims)
