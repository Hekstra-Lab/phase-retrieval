import numpy as np
from keras.models import *
from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Lambda,Subtract


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

dncnn.compile(optimizer="adam",loss="mean_squared_error")

for epoch in range(N_epochs):
    print("Epoch {}".format(epoch))
    for i in range(N_files):
        imgs = np.load("gaussian_data/imgs{}.npy".format(i))
        noisy = np.load("gaussian_data/noisy{}.npy".format(i))
        tensor_noisy = noisy.reshape(noisy.shape+(1,))
        tensor_out = imgs.reshape(imgs.shape+(1,))
        dncnn.fit(tensor_noisy,tensor_out,epochs=1,verbose=False)


dncnn.save("dncnn_50k.h5")
