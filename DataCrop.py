import numpy as np
from random import choice

def modCrop(imgs, modulo):
    if imgs.shape[2] == 1:
        sz = imgs.shape
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1]]

    else:
        sz = imgs.shape[:2];
        sz = sz - np.mod(sz, modulo)
        imgs = imgs[:sz[0], :sz[1], :]

    return imgs

def randomCrop(imgs, size):
    #randomly crop a number of images of same size to be in traning set of the model
    #Args
    #imgs : the img for cropping
    #size : size of each crop, as size * size

    img_xrange = range(imgs.shape[0] - size)
    img_yrange = range(imgs.shape[1] - size)
    start_x = choice(img_xrange)
    start_y = choice(img_yrange)

    return imgs[start_x:start_x+size, start_y:start_y+size]
