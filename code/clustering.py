# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 14:45:12 2023

@author: Ying Tu
"""

#Import essential packages
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

#Get data root
parent = os.path.dirname(os.getcwd())
datadir = parent+'/data/'

#Read data
img1 = cv2.imread(datadir+'Fig1_1939.jpeg') #Image 1 (1939)
img2 = cv2.imread(datadir+'Fig2_2019.jpeg') #Image 2 (2019)

#Define function for KMeans clustering
def KMeans(img,k):
    pixel_values = np.float32(img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    ret, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    m,n,d = img.shape
    segmented_image = labels.flatten().reshape(m,n)
    return segmented_image

#Perform KMeans clustering with k value ranging from 2 to 10
for k in range(2,13):
    
    cluster1 = KMeans(img1,k)
    cluster2 = KMeans(img2,k)
    
    #Save results
    #cv2.imwrite(datadir+'/K-Means/'+'Image-1939-Clusters'+str(k)+'.tif', cluster1)
    #cv2.imwrite(datadir+'/K-Means/'+'Image-2019-Clusters'+str(k)+'.tif', cluster2)
    
    #Plot results
    fig,ax = plt.subplots(1,1,dpi=300)
    plt.imshow(cluster1,cmap='Set1')
    plt.title('Image-1939, Clusters='+str(k))
    outdir = parent+'/result/kmeans_clustering/'
    #plt.savefig(outdir+'Image-1939-Clusters'+str(k)+'.jpeg', bbox_inches = 'tight')
    plt.show()
    
    fig,ax = plt.subplots(1,1,dpi=300)
    plt.imshow(cluster2,cmap='Set1')
    plt.title('Image-2019, $\it{k}$='+str(k))
    #plt.savefig(outdir+'Image-2019-Clusters'+str(k)+'.jpeg', bbox_inches = 'tight')
    plt.show()

