# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 11:04:57 2023

@author: Ying Tu
"""

import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import json
from matplotlib import colors
import matplotlib.patches as mpatches

#Get data root
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))

#Read original images
img2 = cv2.imread(data_root+'/data/Fig2_2019.jpeg')
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

#Read prediction results
input_size=32
pred2rgb = cv2.imread(data_root+"/data/img2_rgb_size"+str(input_size)+"/predict.tif")
pred2gray = cv2.imread(data_root+"/data/img2_gray_size"+str(input_size)+"/predict.tif")
pred1gray = cv2.imread(data_root+"/data/img2_gray_size"+str(input_size)+"/predict_img1.tif") 
pred2rgb  = cv2.cvtColor(pred2rgb , cv2.COLOR_BGR2GRAY)
pred2gray = cv2.cvtColor(pred2gray, cv2.COLOR_BGR2GRAY)
pred1gray = cv2.cvtColor(pred1gray, cv2.COLOR_BGR2GRAY)

#Load model accuracy
acc_json_path1 = os.path.join(data_root,'code/deep_learning/alexnet/accuracy.json')
with open(acc_json_path1, "r") as f:
    acc_indict1 = json.load(f)
accuracy1= acc_indict1['accuracy']

acc_json_path2 = os.path.join(data_root,'code/deep_learning/alexnet_grayscale/accuracy.json')
with open(acc_json_path2, "r") as f:
    acc_indict2 = json.load(f)
accuracy2= acc_indict2['accuracy']

#Load classfication system
class_json_path = os.path.join(data_root,'code/deep_learning/alexnet/class_indices.json')
with open(class_json_path, "r") as f:
    class_indict = json.load(f)

#Set color for each class
color_dict = {
    "class1": "darkgoldenrod",
    "class2": "darkgrey",
    "class3": "white",
    "class4": "green",
    "class5": "brown",
}
colorlist = []
for i in range(0,len(class_indict)):
    classname = class_indict[str(i)]
    classcolor = color_dict[classname]  
    colorlist.append(classcolor)     
cmap = colors.ListedColormap(colorlist)

#Set output filedir
outdir = data_root+'/result/deep_learning/'

sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)

#Visualize results
# Alexnet-2019
fig,ax = plt.subplots(1,1,dpi=300)
ax.imshow(pred2rgb,cmap=cmap)
ax.set_title('AlexNet (%.2f'%(accuracy1)+')') 
#plt.axis('off')
#plt.savefig(outdir+'classfication_maps_alexnet_2019.jpeg', bbox_inches = 'tight')
plt.show()

# Alexnet-grayscale-2019
fig,ax = plt.subplots(1,1,dpi=300)
ax.imshow(pred2gray,cmap=cmap)
ax.set_title('AlexNet (%.2f'%(accuracy2)+')') 
#plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.axis('off')
#plt.savefig(outdir+'classfication_maps_alexnet_2019_grayscale.jpeg', bbox_inches = 'tight')
plt.show()

# Alexnet-grayscale-1939
fig,ax = plt.subplots(1,1,dpi=300)
ax.imshow(pred1gray,cmap=cmap)
plt.axis('off')
#plt.savefig(outdir+'classfication_maps_alexnet_1939_grayscale.jpeg', bbox_inches = 'tight')
plt.show()


fig,axs = plt.subplots(2,2,figsize=(10,6),dpi=300)
axs[0][0].imshow(img2rgb)
axs[0][1].imshow(pred2rgb,cmap=cmap)
axs[1][0].imshow(img2gray,cmap='gray')
axs[1][1].imshow(pred2gray,cmap=cmap)
axs[0][0].set_title('Image-2019 (rgb)')
axs[0][1].set_title('Prediction (rgb)') 
axs[1][0].set_title('Image-2019 (grayscale)')
axs[1][1].set_title('Prediction (grayscale)')
# axs[0][0].axis('off')
# axs[0][1].axis('off')
# axs[1][0].axis('off')
# axs[1][1].axis('off')
plt.subplots_adjust(wspace=0.1, hspace=0.25)
#plt.savefig(outdir+'classfication_maps_alexnet_2019_comparison.jpeg', bbox_inches = 'tight')
plt.show()


# Change map
fig,ax = plt.subplots(1,1,dpi=300)
change_color = 'tomato'
ax.imshow(pred2rgb,cmap=colors.ListedColormap(['k',change_color,change_color,change_color,'k']))
#plt.axis('off')
plt.legend()
patchList = []
legend_dict = {'Changed': change_color, 'Unchanged': 'k'}
for key in legend_dict:
        data_key = mpatches.Patch(color=legend_dict[key], label=key)
        patchList.append(data_key)
plt.legend(handles=patchList,loc='lower right',framealpha=1,fontsize=14)
#plt.savefig(outdir+'change_map_alexnet_2019.jpeg', bbox_inches = 'tight')
plt.show()