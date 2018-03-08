#!usr/bin/python

import numpy as np
from scipy.stats import multivariate_normal

def overlap(x1,x2,r):
    if np.any(np.linalg.norm(x1-x2,axis=1) < r):
        return True
    else:
        return False

def simple_gaussians(N, r, size=100):
    """ Create an image containing N non-overlapping gaussians each with covariance matrices proportional
        to r*I. Default sixe is 100x100"""
    uc = np.zeros((size,size))
    X,Y = np.meshgrid(np.arange(size),np.arange(size))
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X; pos[:, :, 1] = Y
    means = np.zeros((N,2))
    for i in range(N):
        while True:
            test = np.random.randint(int(r/2),int(100-r/2),size=2) #Dont let the gaussian go over the edges
            if not overlap(means,test,r):
                break
        means[i]=test
        mvn = multivariate_normal(mean=test,cov = r)
        uc = uc + mvn.pdf(pos)
    #print means
    return uc

for N in range(1,10):
    mags = np.zeros((1000,100,100))
    ims = np.zeros((1000,100,100))
    for i in range(1000):
        n = np.random.randint(1,51)
        ims[i] = simple_gaussians(n,10)
        mags[i] = np.abs(np.fft.fft2(ims[i]))
    np.save("simple_gaussian_data/mags{}.npy".format(N),mags)
    np.save("simple_gaussian_data/imgs{}.npy".format(N),ims)
