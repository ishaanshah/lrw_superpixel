""" Script to test LRW Superpixel segmentation
    on the Berkeley Segmentation dataset (BSDS300)
"""
import cv2
import matplotlib.pyplot as plt
import os
import math
import numpy as np

from tqdm import tqdm
from lrw import generate_seeds, energy_opt
from utils import im2double, seg2bmap

Nsp = 200  # num of seeds
Thres = 1.35  # Threshold for split
beta = 30  # Gaussian parameter
alpha = 0.9992  # Lazy parameter
nItrs_max = 10  # Limit for the number of iterations

dataset_path = "../images/BSDS300/images/train/"
output_path = "../outputs/BSDS300/train/"

for filename in tqdm(os.listdir(dataset_path)):
    path = os.path.join(dataset_path, filename)
    Nsp = 200  # num of sp
    Thres = 1.35  # threshold for split
    beta = 30  # gaussian parameter
    alpha = 0.9992  # Lazy parameter
    nItrs_max = 10  # limit for the number of iterations
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    scale_percent = 40
    width = math.ceil(img.shape[1] * scale_percent / 100)
    height = math.ceil(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    gray_img = cv2.cvtColor(img.astype("uint8"), cv2.COLOR_RGB2GRAY)
    orig_img = img.copy()
    img = im2double(img)
    X, Y, Z = img.shape

    seeds = generate_seeds(Nsp, im2double(gray_img / 255))

    res, seeds, iters = energy_opt(orig_img, seeds, alpha, Nsp, nItrs_max, beta, Thres)
    bmap = seg2bmap(res, Y, X)
    idx = np.nonzero(bmap > 0)

    bmapOnImg = img
    temp = img[:, :, 0]
    temp[idx] = 0
    bmapOnImg[:, :, 0] = temp
    if Z == 3:
        temp = img[:, :, 1]
        temp[idx] = 1
        bmapOnImg[:, :, 1] = temp
        temp = img[:, :, 2]
        temp[idx] = 1
        bmapOnImg[:, :, 2] = temp

    plt.imshow(bmapOnImg)
    plt.axis("off")
    seeds_x = [i[0] for i in seeds]
    seeds_y = [i[1] for i in seeds]
    plt.savefig(output_path + filename)
    plt.close()
