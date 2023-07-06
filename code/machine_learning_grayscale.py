# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:34:17 2023

@author: Ying Tu
"""

#Import essential packages
import os
import cv2
import fast_glcm
from dbfread import DBF
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')


#Function to compute textures
def computeTextures(img):
    asm, ene = fast_glcm.fast_glcm_ASM(img,ks=5)
    cont = fast_glcm.fast_glcm_contrast(img,ks=5)
    diss = fast_glcm.fast_glcm_dissimilarity(img,ks=5)
    homo = fast_glcm.fast_glcm_homogeneity(img,ks=5)
    ent = fast_glcm.fast_glcm_entropy(img,ks=5)
    std = fast_glcm.fast_glcm_std(img,ks=5)  
    
    textures = np.concatenate((np.expand_dims(img,axis=-1),
                               np.expand_dims(asm,axis=-1),
                               np.expand_dims(cont,axis=-1),
                               np.expand_dims(diss,axis=-1),
                               np.expand_dims(ent,axis=-1),
                               np.expand_dims(ene,axis=-1),
                               np.expand_dims(homo,axis=-1),
                               np.expand_dims(std,axis=-1)), axis=2)

    return textures

#Function to plot textures
def plotTextures(img,outname):
    grays = img[:,:,0]
    asm = img[:,:,1]
    cont = img[:,:,2]
    diss = img[:,:,3]
    ent = img[:,:,4]
    ene = img[:,:,5]
    homo = img[:,:,6]
    std = img[:,:,7]
    
    row = 2
    col = 4
    fs = 10
    
    plt.figure(figsize=(12,4),dpi=300)
        
    plt.subplot(row,col,1)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(grays,cmap='gray')
    plt.title('original', fontsize=fs)

    plt.subplot(row,col,2)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(std,cmap='gray')
    plt.title('std', fontsize=fs)

    plt.subplot(row,col,3)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(cont,cmap='gray')
    plt.title('contrast', fontsize=fs)

    plt.subplot(row,col,4)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(diss,cmap='gray')
    plt.title('dissimilarity', fontsize=fs)

    plt.subplot(row,col,5)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(homo,cmap='gray')
    plt.title('homogeneity', fontsize=fs)

    plt.subplot(row,col,6)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(asm,cmap='gray')
    plt.title('asm', fontsize=fs)

    plt.subplot(row,col,7)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(ene,cmap='gray')
    plt.title('energy', fontsize=fs)

    plt.subplot(row,col,8)
    plt.tick_params(labelbottom=False, labelleft=False, left=False, bottom=False)
    plt.imshow(ent,cmap='gray')
    plt.title('entropy', fontsize=fs)
    
    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    outdir = parent+'/result/machine_learning/'
    plt.savefig(outdir+outname, bbox_inches='tight')
    plt.show()
    
#Get data root
parent = os.path.dirname(os.getcwd())
datadir = parent+'/data/'

#Read images
img1 = cv2.imread(datadir+'Fig1_1939.jpeg')
img2 = cv2.imread(datadir+'Fig2_2019.jpeg')
img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

img1texr = computeTextures(img1gray)
img2texr = computeTextures(img2gray)
#plotTextures(img1texr,'texture_1939_grayscale.jpeg')
plotTextures(img2texr,'texture_2019_grayscale.jpeg')

#Whether to use texture features
useTextures = True #False

if useTextures:
    feature_name = ['Value','ASM','Contrast','Dissimilarity','Entropy','Energy','Homogeneity','Variance']
    X_all1 = img1texr.reshape((-1, len(feature_name)))
    X_all2 = img2texr.reshape((-1, len(feature_name)))
else:
    feature_name = ['Value']
    X_all1 = img1gray.reshape((-1, len(feature_name)))
    X_all2 = img2gray.reshape((-1, len(feature_name)))    


#Read sample data
dfsample = pd.DataFrame(DBF(datadir+'sample/sample_2019_location.dbf'))

#Extract grayscale value of img2 for each sample point
values = dfsample[['Label','Row','Column']].values
m,n = values.shape
res = []
for i in range(0,m):
    label = values[i][0]
    row = values[i][1]
    col = values[i][2]
    b = img2texr[row][col][0]
    asm = img2texr[row][col][1]
    cont = img2texr[row][col][2]
    diss = img2texr[row][col][3]
    ent = img2texr[row][col][4]
    ene = img2texr[row][col][5]
    homo = img2texr[row][col][6]
    std = img2texr[row][col][7]
    
    res.append([label,row,col,b,asm,cont,diss,ent,ene,homo,std])

#Create a new dataframe - dfsample2
dfsample2 = pd.DataFrame(res,columns=['Label','Row','Column','Value',
                                      'ASM','Contrast','Dissimilarity',
                                      'Entropy','Energy','Homogeneity','Variance'])

#Split sample data into features (X) and target (y)
X = dfsample2[feature_name]#.values
y = dfsample2['Label']#.values

#Split sample data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Define machine learning models
modelnames = ['lda','nb','knn','rf','gbc']
modelnamesfull = ['Linear Discriminant Analysis','Gaussian Naive Bayes','K-nearest Neighbors','Random Forest','Gradient Boosting Tree']
models = []
models.append(LinearDiscriminantAnalysis())
models.append(GaussianNB())
models.append(KNeighborsClassifier())
models.append(RandomForestClassifier())
models.append(GradientBoostingClassifier())

#Hyperparameter tuning
param_dist = {'lda': {'solver': ['svd', 'lsqr', 'eigen']},
              'nb': {'var_smoothing': np.logspace(0,-9, num=100)},
              'knn': {'n_neighbors': randint(2,50)},
              'rf': {'n_estimators': randint(50,500),'max_depth': randint(1,20), 'min_samples_split': randint(2,10),'min_samples_leaf': randint(2,10)},
              'gbc': {'n_estimators': randint(50,500),'max_depth': randint(1,20), 'min_samples_split': randint(2,10),'min_samples_leaf': randint(2,10)},
              }

#Perform supervised classification for each model
acclist = []
for i in range(0,len(modelnames)):

    model = models[i]
    modelname = modelnames[i]
    modelnamefull = modelnamesfull[i]
    
    # Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(model, 
                                     param_distributions = param_dist[modelname], 
                                     n_iter=10, 
                                     cv=5
                                     )
    
    # Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    
    # Create a variable for the best model
    best_model = rand_search.best_estimator_
    
    # Generate predictions with the best model
    y_pred = best_model.predict(X_test)
    
    # Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #Calculate classification accuracy with test data
    accuracy = accuracy_score(y_test, y_pred)
    
    #Append accuracy to the list
    acclist.append(accuracy)
    
    print (modelnamefull)
    print ("Model:",best_model)    
    print ("Accuracy: %.2f"%(accuracy))
    print ()
    
    #Predict land covers for img1 and img2
    y_all1 = best_model.predict(X_all1)
    y_all2 = best_model.predict(X_all2)
    
    #Concatenate the results to new arrays
    m,n,d = img2.shape
    img_pred1 = y_all1.reshape(m,n)
    img_pred2 = y_all2.reshape(m,n)
    
    if i==0:
        img_pred_all1 = np.expand_dims(img_pred1, axis=-1)
        img_pred_all2 = np.expand_dims(img_pred2, axis=-1)
        cm_all = np.expand_dims(cm, axis=-1)
    else:
        img_pred_all1 = np.concatenate((img_pred_all1, np.expand_dims(img_pred1, axis=-1)), axis=2)
        img_pred_all2 = np.concatenate((img_pred_all2, np.expand_dims(img_pred2, axis=-1)), axis=2)
        cm_all = np.concatenate((cm_all, np.expand_dims(cm, axis=-1)), axis=2)

outdir = parent+'/result/machine_learning/'

#Plot confusion matrix
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,axs = plt.subplots(2,3,figsize=(15,9), dpi=300)
ConfusionMatrixDisplay(confusion_matrix=cm_all[:,:,0]).plot(ax=axs[0][0])
ConfusionMatrixDisplay(confusion_matrix=cm_all[:,:,1]).plot(ax=axs[0][1])
ConfusionMatrixDisplay(confusion_matrix=cm_all[:,:,2]).plot(ax=axs[0][2])
ConfusionMatrixDisplay(confusion_matrix=cm_all[:,:,3]).plot(ax=axs[1][0])
ConfusionMatrixDisplay(confusion_matrix=cm_all[:,:,4]).plot(ax=axs[1][1])

axs[0][0].set_title(modelnamesfull[0]+' (%.2f'%(acclist[0])+')') 
axs[0][1].set_title(modelnamesfull[1]+' (%.2f'%(acclist[1])+')') 
axs[0][2].set_title(modelnamesfull[2]+' (%.2f'%(acclist[2])+')') 
axs[1][0].set_title(modelnamesfull[3]+' (%.2f'%(acclist[3])+')') 
axs[1][1].set_title(modelnamesfull[4]+' (%.2f'%(acclist[4])+')')
axs[1][2].remove()
#plt.savefig(outdir+'confusion_matrix_2019_grayscale.jpeg', bbox_inches='tight')
plt.show()
        
#Plot classification maps   
# 2019
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,axs = plt.subplots(2,3,figsize=(18,8), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
axs[0][0].imshow(img2gray,cmap='gray')
axs[0][1].imshow(img_pred_all2[:,:,0],cmap=cmap)
axs[0][2].imshow(img_pred_all2[:,:,1],cmap=cmap)
axs[1][0].imshow(img_pred_all2[:,:,2],cmap=cmap)
axs[1][1].imshow(img_pred_all2[:,:,3],cmap=cmap)
axs[1][2].imshow(img_pred_all2[:,:,4],cmap=cmap)
axs[0][0].set_title('Satellite image in 2019')
axs[0][1].set_title(modelnamesfull[0]+' (%.2f'%(acclist[0])+')') 
axs[0][2].set_title(modelnamesfull[1]+' (%.2f'%(acclist[1])+')') 
axs[1][0].set_title(modelnamesfull[2]+' (%.2f'%(acclist[2])+')') 
axs[1][1].set_title(modelnamesfull[3]+' (%.2f'%(acclist[3])+')') 
axs[1][2].set_title(modelnamesfull[4]+' (%.2f'%(acclist[4])+')')
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_comparison_2019_grayscale.jpeg', bbox_inches='tight')
plt.show()
    
# 1939 
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,axs = plt.subplots(2,3,figsize=(18,8), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
axs[0][0].imshow(img1gray,cmap='gray')
axs[0][1].imshow(img_pred_all1[:,:,0],cmap=cmap)
axs[0][2].imshow(img_pred_all1[:,:,1],cmap=cmap)
axs[1][0].imshow(img_pred_all1[:,:,2],cmap=cmap)
axs[1][1].imshow(img_pred_all1[:,:,3],cmap=cmap)
axs[1][2].imshow(img_pred_all1[:,:,4],cmap=cmap)
axs[0][0].set_title('Satellite image in 1939')
axs[0][1].set_title(modelnamesfull[0]) 
axs[0][2].set_title(modelnamesfull[1]) 
axs[1][0].set_title(modelnamesfull[2]) 
axs[1][1].set_title(modelnamesfull[3]) 
axs[1][2].set_title(modelnamesfull[4])
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_comparison_1939_grayscale.jpeg', bbox_inches='tight')
plt.show()


#Integrate classification maps of different models through voting
img_pred_vote1 = img_pred1
img_pred_vote2 = img_pred2
for i in range (0,m):
    for j in range(0,n):
        nums1 = img_pred_all1[i,j,:]
        img_pred_vote1[i][j] = np.argmax(np.bincount(nums1)) #mode value
        
        nums2 = img_pred_all2[i,j,:]
        img_pred_vote2[i][j] = np.argmax(np.bincount(nums2)) #mode value

#Compute accuracy for ensemble learning
values = dfsample2[['Row','Column']].values
m,n = dfsample2[['Row','Column']].shape
sample_pred = []
for i in range(0,m):
    row = values[i][0] #row
    col = values[i][1] #column
    label_pred = img_pred_vote2[row][col]
    sample_pred.append(label_pred)
dfsample2['ensemble_pred'] = sample_pred
dfsample2test = dfsample2.clip(y_test, axis=0).dropna()
y_test = dfsample2test['Label']
y_pred = dfsample2test['ensemble_pred']
accuracy = accuracy_score(y_test, y_pred)
print ("Ensemble learning")
print ("Accuracy: %.2f"%(accuracy))


#Plot ensemble learning results
# 2019
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
ax1.imshow(img2gray,cmap='gray')
ax2.imshow(img_pred_vote2,cmap=cmap)
ax1.set_title('Satellite image in 2019')
ax2.set_title('Ensemble learning (%.2f'%(accuracy)+')') 
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_ensemble_2019_grayscale.jpeg', bbox_inches = 'tight')
plt.show()

# 1939
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
ax1.imshow(img1gray,cmap='gray')
ax2.imshow(img_pred_vote1,cmap=cmap)
ax1.set_title('Satellite image in 1939')
ax2.set_title('Ensemble learning') 
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_ensemble_1939_grayscale.jpeg', bbox_inches = 'tight')
plt.show()
