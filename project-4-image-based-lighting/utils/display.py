# -*- coding: utf-8 -*-
""" Contains displaying of image / hdr images """

# imports
import cv2
import numpy as np
import matplotlib.pyplot as plt

# metadata
__author__ = "Jae Yong Lee"
__copyright__ = "Copyright 2019, CS445"
__credits__ = ["Jae Yong Lee"]
__license__ = "None"
__version__ = "1.0.0"
__maintainer__ = "Jae Yong Lee"
__email__ = "lee896@illinois.edu"
__status__ = 'production'

def display_hdr_image_linear(hdr_image: np.ndarray):
    '''
    Given HDR image, display by linear scale
    
    Args:
      - hdr_image: HxWxC HDR float32 image
    '''
    hmin = hdr_image[hdr_image == hdr_image].min()
    hmax = hdr_image[hdr_image == hdr_image].max()
    rescaled = (hdr_image - hmin) / (hmax - hmin)
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(rescaled[:, :])
    
def display_hdr_image(hdr_image: np.ndarray):
    '''
    Given HDR image, display by tonemapping
    
    Args:
      - hdr_image: HxWxC HDR float32 image
    '''
    tonemapper = cv2.createTonemapDrago(1.0, 0.7)
    tonemapped = tonemapper.process(hdr_image)
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(tonemapped[:, :])
    
    
def _rescale_log_irradicance(le):
    '''
    Helper function to rescale log irradiance in visible range
    '''
    le_min = le[le != -float('inf')].min()
    le_max = le[le != float('inf')].max()
    
    le = (le - le_min) / (le_max - le_min)                
    mask = (le == float('inf')) | (le == -float('inf'))

    le[mask] =0
    return le

def display_log_irradiances(log_irradiances: np.ndarray):
    '''
    Given Log irradiances, display by rescaling
    
    Args:
      - log_irradiances: NxHxWxC HDR float32 image
    '''
    N, H, W, C = log_irradiances.shape
    assert N == 3
    
    fix, axes = plt.subplots(1, 3)
    [a.axis('off') for a in axes.ravel()]
    for n in range(N):
        axes[n].imshow(_rescale_log_irradicance(log_irradiances[n]))


    