# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 11:06:04 2023

@author: Ying Tu
"""

#Import essential packages
import os
import cv2
import matplotlib.pyplot as plt

#Get data root
parent = os.path.dirname(os.getcwd())
datadir = parent+'/data/'

#Read data
img1 = cv2.imread(datadir+'Fig1_1939.jpeg') #Image 1 (1939)
img2 = cv2.imread(datadir+'Fig2_2019.jpeg') #Image 2 (2019)
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) #Convert RGB channels of image2
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #Convert image2 to grayscale
#cv2.imwrite(parent+'\\data\\Fig2_2019_grayscale.jpeg', img2gray)

#Plot images
fig,ax = plt.subplots(1,1,dpi=300)
plt.imshow(img1)
plt.title('Image-1939')
plt.show()

fig,ax = plt.subplots(1,1,dpi=300)
plt.imshow(img2rgb)
plt.title('Image-2019')
plt.show()

fig,ax = plt.subplots(1,1,dpi=300)
plt.imshow(img2gray,cmap='gray')
plt.title('Image-2019-grayscale')
plt.show()
