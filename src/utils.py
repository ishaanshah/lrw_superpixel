import numpy as np
from scipy import sparse
import pyrtools as pt
import math
import matplotlib.pyplot as plt
import networkx


def im2double(im):
    #     info = np.iinfo(im.dtype) # Get the data type of the input image
    #     return im.astype(np.float) / info.max # Divide all values by the largest possible value in the datatype
    min_val = np.min(im)
    max_val = np.max(im)
    out = (im.astype("float") - min_val) / (max_val - min_val)
    return out


def gauss2D(shape, sigma):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]
    h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def local_min(img, prhs):
    iWidth, iHeight = img.shape
    plhs = [0, 0]
    im = img.flatten()
    plhs[0] = np.zeros((iWidth, iHeight))
    plhs[1] = np.zeros((iWidth, iHeight))
    N = prhs

    for j in range(N, iHeight - N):
        for i in range(N, iWidth - N):
            dMinValue = np.float("Inf")
            for k in range(-N, N + 1):
                for l in range(-N, N + 1):
                    iIndex = (j + k) * iWidth + i + l
                    if im[iIndex] < dMinValue:
                        dMinValue = im[iIndex]
                        plhs[0][i, j] = i + l + 1
                        plhs[1][i, j] = j + k + 1
    return plhs[0], plhs[1]


def get_weights(image, sigma):
    """Generate a adjacency matrix with weights as described in the paper

    Args:
        image: The input image (LAB)
        sigma: Gaussian parameter
    Returns:
        adj: The adjacency matrix
    """
    height, width = image.shape[:2]
    size = height * width
    flat_image = image.reshape(size, -1).astype(dtype="float")

    # Get edges
    graph = networkx.lattice.grid_2d_graph(height, width)
    graph.add_edges_from([
        ((x, y), (x+1, y+1))
        for x in range(height-1)
        for y in range(width-1)
    ] + [
        ((x+1, y), (x, y+1))
        for x in range(height-1)
        for y in range(width-1)
    ])
    edges = []
    for edge in graph.edges:
        edges.append([edge[0][0]*width+edge[0][1], edge[1][0]*width+edge[1][1]])
    edges = np.array(edges)

    weights = np.sqrt(
        np.sum((flat_image[edges[:, 0]] - flat_image[edges[:, 1]]) ** 2, axis=1)
    )

    # Normalize to 0-1
    mx = weights.max()
    mn = weights.min()
    if mx == mn:
        weights[:] = 1
    weights = (weights - mn) / (mx - mn)
    # Perform gaussian
    weights = np.exp(-sigma * weights) + 1e-5

    adj = sparse.coo_matrix(
        (
            np.concatenate((weights, weights)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(size, size),
    )

    return adj

def seg2bmap(seg,width=None,height=None):
    """
    From a segmentation, compute a binary boundary map with 1 pixel wide
    boundaries.  The boundary pixels are offset by 1/2 pixel towards the
    origin from the actual segment boundary.
    Arguments:
        seg     : Segments labeled from 1..k.
        width	  :	Width of desired bmap  <= seg.shape[1]
        height  :	Height of desired bmap <= seg.shape[0]
    Returns:
        bmap (ndarray):	Binary boundary map.
        David Martin <dmartin@eecs.berkeley.edu>
        January 2003
 """
#     seg = seg.astype(np.bool)
#     seg[seg>0] = 1
    assert np.atleast_3d(seg).shape[2] == 1
    width  = seg.shape[1] if width  is None else width
    height = seg.shape[0] if height is None else height
    h,w = seg.shape[:2]
    ar1 = float(width) / float(height)
    ar2 = float(w) / float(h)
    
    assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
            'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

    e  = np.zeros_like(seg)
    s  = np.zeros_like(seg)
    se = np.zeros_like(seg)

    e[:,:-1]    = seg[:,1:]
    s[:-1,:]    = seg[1:,:]
    se[:-1,:-1] = seg[1:,1:]

    b        = seg^e | seg^s | seg^se
    b[-1,:]  = seg[-1,:]^e[-1,:]
    b[:,-1]  = seg[:,-1]^s[:,-1]
    b[-1,-1] = 0

    if w == width and h == height:
        bmap = b
    else:
        bmap = np.zeros((height,width))
        for x in range(w):
            for y in range(h):
                if b[y,x]:
                    j = 1+floor((y-1)+height / h)
                    i = 1+floor((x-1)+width  / h)
                    bmap[j,i] = 1;

    return bmap
