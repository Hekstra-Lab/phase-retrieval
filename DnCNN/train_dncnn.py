"""
Train DnCNN with on the fly data generation. Images generated will be 128x128 images containing between 1 and 50 non-overlapping gaussians with variance 10. 

Run as python train_dncnn.py net_name N_files batch_size

"""


import numpy as np
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract
from keras.callbacks import ModelCheckpoint
import argparse
import os
from scipy.stats import multivariate_normal

parser = argparse.ArgumentParser(description = "Train DnCNN with on the fly data generation")
parser.add_argument("net_name", type=str, help="Name to save the network")
parser.add_argument("N_epochs", type=int, help="Number of epochs to train")
parser.add_argument("N_files", type=int, help="Number of files to use per epoch") 
parser.add_argument("--batch_size", default = 32, type=int, help="Training batch size. N_files should be a multiple of batch_size")
parser.add_argument("--save_network", default = True, type=bool, help="Save model after training is complete. If true model will be saved as net_name.h5")
parser.add_argument("--noise_level", metavar= "sigma", default = 0.2, type = float, help="Variance of additive gaussin noise.")

args = parser.parse_args()

print(args.net_name)

print(args.N_epochs)



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


def DnCNN():
    
    inpt = Input(shape=(None,None,1))
    # 1st layer, Conv+relu
    x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(inpt)
    x = Activation('relu')(x)
    # 15 layers, Conv+BN+relu
    for i in range(15):
        x = Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same')(x)
        x = BatchNormalization(axis=-1, epsilon=1e-3)(x)
        x = Activation('relu')(x)   
    # last layer, Conv
    x = Conv2D(filters=1, kernel_size=(3,3), strides=(1,1), padding='same')(x)
    x = Subtract()([inpt, x])   # input - noise
    model = Model(inputs=inpt, outputs=x)
    
    return model

N_epochs=25
N_files = 50

dncnn = DnCNN()

print ("Made it here")

dncnn.compile(optimizer="adam",loss="mean_squared_error")

for epoch in range(args.N_epochs):
    print("Epoch {}".format(epoch))
    imgs = np.zeros((args.N_files,128,128))
    noise = np.random.normal(0,args.noise_level,size=imgs.shape)
    for i in range(args.N_files):
        n = np.random.randint(1,51)
        imgs[i] = simple_gaussians(n,10)
    noisy = (imgs+noise).reshape(imgs.shape + (1,))
    out = imgs.reshape(imgs.shape+(1,))
    dncnn.fit(noisy,out,epochs=1,batch_size=args.batch_size,verbose=False)


dncnn.save(args.net_name+".h5")
