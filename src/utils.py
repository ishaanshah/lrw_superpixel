import numpy as np
import pyrtools as pt
import math
import cv2

def im2double(im):
#     info = np.iinfo(im.dtype) # Get the data type of the input image
#     return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype
    min_val = np.min(im)
    max_val = np.max(im)
    out = (im.astype('float')-min_val) / (max_val-min_val)
    return out

def gauss2D(shape,sigma):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def local_min(img,prhs):
    iWidth, iHeight = img.shape
    plhs = [0,0]
    im = img.flatten()
    plhs[0] = np.zeros((iWidth,iHeight))
    plhs[1] = np.zeros((iWidth,iHeight))
    N = prhs
    
    for j in range(N,iHeight-N):
        for i in range(N,iWidth-N):
            dMinValue = np.float("Inf")
            for k in range(-N,N+1):
                for l in range(-N,N+1):
                    iIndex = (j+k)*iWidth+i+l
                    if (im[iIndex] < dMinValue):
                        dMinValue = im[iIndex]
                        plhs[0][i,j] = i+l+1
                        plhs[1][i,j] = j+k+1
    return plhs[0],plhs[1]