import numpy as np
import pyrtools as pt
import math
import cv2


def energy_opt(image, seeds, alpha, count, iters, sigma, thres):
    """Find and optimize the superpixel results
    Args:
        image: Original Image (RGB / Grayscale)
        seeds: The initial seed positions
        alpha: Probability of staying put in LRW
        count: Number of superpixels
        iters: Max number of iterations
        sigma: Gaussian parameter
        thres: Threshold to split bigger superpixels
    Returns:
        label: Labeled image
        seeds: The optimized seed positions
        iters: The number of iterations taken for convergence
    """

    height, width = image.shape[:2]
    if len(image.shape) > 2:
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
