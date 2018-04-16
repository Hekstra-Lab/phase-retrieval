import sys
sys.path.append("..")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context("notebook", font_scale=1.3)
import numpy as np
import pandas as pd
import PhaseRetrieval
import simple_gaussian_data_generator
from skimage import data
from skimage import color, img_as_float
from skimage.feature import register_translation
import os
import glob

datasets = glob.glob("dataset*.npy")

for dataset in datasets:
    data = np.load(dataset)
    size = len(data)

    errs = np.zeros(size)
    X = []
    Y = []
    success = []
    for i, img in enumerate(data):
        if i%250==0:
            print "Iteration %d" %i
        
        mags = np.abs(np.fft.fftn(img))
        chio = PhaseRetrieval.PhaseRetrieval(mags)
        chio.CHIO(n_iter=250)
        chio.calc_real_space_error(img, plot=False)
        errs[i] = chio.real_space_err_track[-1]
        if chio.real_space_err_track[-1] < 0.5:
            diffs = np.where(np.diff(chio.real_space_err_track) < -0.05)[0]
            if len(diffs) > 0 and diffs[0] != 0:
                X.append(chio.ndm_track[diffs[0]-1])
                Y.append(chio.rs_track[diffs[-1]+1])
            success.append(i)
        del chio
    np.save("X%s.npy" %dataset[-8:-4], np.array(X))
    np.save("Y%s.npy" %dataset[-8:-4], np.array(Y))
    np.save("errors%s.npy" %dataset[-8:-4], errs)
    print dataset, sum(success)/float(len(dataset))
    
        
