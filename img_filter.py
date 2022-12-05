# -*-coding:utf-8-*-
"""
File Name: image_deeplearning.py
Date: 2022/12/5
Create File By Author: Bingqi Shen
"""
import cv2
import numpy as np
import random

def clamp(pv):
    if pv > 255:
        return 255
    if pv < 0:
        return 0
    else:
        return pv

# 加高斯噪声
def gaussian_noise(image): 
    image_ori = image.copy()
    
    h, w, c = image.shape
    for row in range(h):
        for col in range(w):
            s = np.random.normal(0, 20, 3)
            # s = np.random.normal(0, 60, 3)
            b = image[row, col, 0] # blue
            g = image[row, col, 1] # green
            r = image[row, col, 2] # red
            image[row, col, 0] = clamp(b + s[0])
            image[row, col, 1] = clamp(g + s[1])
            image[row, col, 2] = clamp(r + s[2])
    cv2.imwrite('noise_kiki.png', image)

    return image

# 加椒盐噪声
def sp_noise(image):
    prob = 0.01
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output


# 均值滤波
def mean_filter(noise_img):
    result = cv2.blur(noise_img, (5, 5))
    cv2.imwrite('mean_filter_kiki.png', result)

# 中值滤波
def median_filter(noise_img):
    result = cv2.medianBlur(noise_img, 5)
    cv2.imwrite("median_filter_kiki.png", result)



def gaussian_filter(noise_img):
    gaussian_blurred = cv2.GaussianBlur(noise_img, (5, 5), 0)
    cv2.imwrite('gaussian_filter_kiki.png', gaussian_blurred)



def bilateral_filter(noise_img):
    bilateral_blurred = cv2.bilateralFilter(noise_img, d=20, sigmaColor=50, sigmaSpace=15)
    cv2.imwrite('bilateral_filter_kiki.png', bilateral_blurred)



if __name__ == '__main__':
    path = 'kiki.png'
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    noise_img = gaussian_noise(img)
    # noise_img = sp_noise(img)
    mean_filter(noise_img)
    median_filter(noise_img)
    gaussian_filter(noise_img)
    bilateral_filter(noise_img)