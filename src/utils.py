import numpy as np
from scipy import sparse
import pyrtools as pt
import math
import cv2


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
    flat_image[:, 0] = flat_image[:, 0] * 100 / 255
    flat_image[:, 1:] -= 128

    # Get edges
    edges = np.concatenate(
        (
            np.vstack((np.arange(size), np.arange(size) + 1)).T,
            np.vstack((np.arange(size), np.arange(size) + width)).T,
        )
    )

    # Filter invalid edges
    edges = np.array(
        [
            edge
            for edge in edges
            if (edge[0] > 0 and edge[0] < size and edge[1] >= 0 and edge[1] < size)
        ]
    )
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
