import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

imageRootDir = './ToExtract/'

for file in os.listdir(imageRootDir):
    if file.endswith('real_A.png'):
        print(file)
        labelImg = cv2.imread(os.path.join(imageRootDir, file), cv2.IMREAD_GRAYSCALE)
        genImg = cv2.imread(os.path.join(imageRootDir, file.replace('real_A.png', 'fake_B.png')), cv2.IMREAD_GRAYSCALE)

        print(labelImg.shape)

        ret, thrLabel = cv2.threshold(labelImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        im2, contours, hierarchy = cv2.findContours(thrLabel, cv2.CV_RETR_TREE, cv2.CV_CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        rect = cv2.boundingRect(cnt)
        print(rect)

        plt.subplot(2, 3, 1)
        plt.imshow(labelImg, 'gray')
        plt.subplot(2, 3, 2)
        plt.imshow(genImg, 'gray')
        plt.subplot(2, 3, 3)
        plt.imshow(thrLabel, 'gray')
        plt.show()
