# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 17:42:12 2023

@author: Ying Tu
"""

import os
import cv2
from dbfread import DBF
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
from shutil import copy, rmtree

def mk_file(file_path: str):
    if os.path.exists(file_path):
        rmtree(file_path)
    os.makedirs(file_path)

def clear_dir(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
            
data_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))

img1 = cv2.imread(data_root+'/data/Fig1_1939.jpeg')
img2 = cv2.imread(data_root+'/data/Fig2_2019.jpeg')
img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img_size = 32

img = img2gray
m,n = img.shape
newrow = m+img_size
newcol = n+img_size
newimg  = np.zeros([newrow,newcol], dtype=np.uint8)
start = int(img_size/2)
newimg[start:start+m,start:start+n] = img

fig,ax = plt.subplots(1,1,dpi=300)
plt.imshow(newimg,cmap='gray')

sample_file = os.path.join(data_root,'data','sample','sample_2019_location.dbf')
dfsample = pd.DataFrame(DBF(sample_file))
class_dict = {
    1: "Cropland",
    2: "Built-up (Low)",
    3: "Built-up (High)",
    4: "Vegetation",
    5: "Bareland"
}    

sampledir = os.path.join(data_root,'data','img2_gray_size'+str(img_size),'sample')
mk_file(sampledir)
clear_dir(sampledir)

for label in range(1,len(class_dict)+1):
    classname = class_dict[label]

    classdir = os.path.join(sampledir,'class'+str(label))
    mk_file(classdir)
    
    dfclass = dfsample.loc[dfsample['Label']==label]
    values = dfclass[['Row','Column']].values
    m,n = values.shape
    for i in range(0,m):
        row = values[i][0]
        col = values[i][1]
        subset = newimg[row:row+img_size,col:col+img_size]
        subset3 = cv2.cvtColor(subset, cv2.COLOR_GRAY2BGR)
        outname = 'row'+str(row)+'_col'+str(col)+'.jpg'
        #cv2.imwrite(os.path.join(classdir,outname), subset3)
        cv2.imwrite(os.path.join(classdir,outname), subset)
print ('Sample data created!')      
