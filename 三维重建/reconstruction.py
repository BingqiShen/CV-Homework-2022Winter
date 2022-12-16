#!/usr/bin/env python
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def sgbm(imgL, imgR):
    T1 = time.time()
    window_size = 1
    min_disp = 0
    num_disp = 320 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
							numDisparities = num_disp,
							blockSize = 5,
							P1 = 8 * 3 * window_size**2,
							P2 = 32 * 3 * window_size**2,
							disp12MaxDiff = 1,
							uniquenessRatio = 10,
							speckleWindowSize = 100,
							speckleRange = 32
							)

    # disparity = stereo.compute(imgL, imgR)
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print((time.time() - T1) * 1000)
    
    plt.imshow(disparity, 'gray')
    plt.savefig('sgbm.png')
    plt.show()

def bm(imgL, imgR):
    T1 = time.time()
    # SAD window size should be between 5..255
    block_size = 11

    min_disp = 0
    num_disp = 320 - min_disp
    uniquenessRatio = 10


    stereo = cv2.StereoBM_create(numDisparities = num_disp, blockSize = block_size)
    stereo.setUniquenessRatio(uniquenessRatio)

    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print((time.time() - T1) * 1000)

    plt.imshow(disparity,'gray')
    plt.savefig('bm.png')
    plt.show()




if __name__ == "__main__":
    print('loading images...')
    imgL = cv2.imread('chess2/im0.png', 0)  
    imgR = cv2.imread('chess2/im1.png', 0)

    bm(imgL, imgR)
    sgbm(imgL, imgR)
