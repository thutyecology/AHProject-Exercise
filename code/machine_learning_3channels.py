# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:34:17 2023

@author: Ying Tu
"""
    
#Import essential packages
import os
import cv2
from dbfread import DBF
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import colors
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
import warnings
warnings.filterwarnings('ignore')

    
#Get data root
parent = os.path.dirname(os.getcwd())
datadir = parent+'/data/'

#Read images
img1 = cv2.imread(datadir+'Fig1_1939.jpeg')
img2 = cv2.imread(datadir+'Fig2_2019.jpeg')
img2rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

#Read sample data
dfsample = pd.DataFrame(DBF(datadir+'sample/sample_2019_location.dbf'))
dfsample.head() #Column 'Label' indicates different land cover types

#Extract RGB values of img2 for each sample point
values = dfsample[['Label','Row','Column']].values
m,n = values.shape
res = []
for i in range(0,m):
    label = values[i][0] #land cover type
    row = values[i][1] #row
    col = values[i][2] #column
    b1 = int(img2[row][col][0]) #b1 value
    b2 = int(img2[row][col][1]) #b2 value
    b3 = int(img2[row][col][2]) #b3 value
    res.append([label,row,col,b1,b2,b3])

#Create a new dataframe - dfsample2
dfsample2 = pd.DataFrame(res,columns=['Label','Row','Column','b1_2019','b2_2019','b3_2019'])

#Split sample data into features (X) and target (y)
X = dfsample2[['b1_2019','b2_2019','b3_2019']]#.values
y = dfsample2['Label']#.values

#Split sample data into training (80%) and test (20%) sets
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
m,n,d = img2.shape
for i in range(0,len(modelnames)):

    model = models[i]
    modelname = modelnames[i]
    modelnamefull = modelnamesfull[i]
    
    #Use random search to find the best hyperparameters
    rand_search = RandomizedSearchCV(model, 
                                     param_distributions = param_dist[modelname], 
                                     n_iter=10, 
                                     cv=5
                                     )
    
    #Fit the random search object to the data
    rand_search.fit(X_train, y_train)
    
    #Create a variable for the best model
    best_model = rand_search.best_estimator_
    
    #Generate predictions with the best model
    y_pred = best_model.predict(X_test)
    
    #Create the confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    #Calculate classification accuracy with test data
    accuracy = accuracy_score(y_test, y_pred)
    
    #Append accuracy to the list
    acclist.append(accuracy)
    
    print (modelnamefull)
    print ("Model:",best_model)    
    print ("Accuracy: %.2f"%(accuracy))
    print ()
    
    #Predict land covers for the whole image
    X_all = img2.reshape((-1, 3))
    y_all = best_model.predict(X_all)
    img2_pred = y_all.reshape(m,n)
    
    #Concatenate the results to new arrays 
    if i==0:
        img2_pred_all = np.expand_dims(img2_pred, axis=-1)
        cm_all = np.expand_dims(cm, axis=-1)
    else:
        img2_pred_all = np.concatenate((img2_pred_all, np.expand_dims(img2_pred, axis=-1)), axis=2)
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
#plt.savefig(outdir+'confusion_matrix_2019.jpeg', bbox_inches='tight')
plt.show()

#Plot classification maps    
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,axs = plt.subplots(2,3,figsize=(18,8), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
axs[0][0].imshow(img2rgb)
axs[0][1].imshow(img2_pred_all[:,:,0],cmap=cmap)
axs[0][2].imshow(img2_pred_all[:,:,1],cmap=cmap)
axs[1][0].imshow(img2_pred_all[:,:,2],cmap=cmap)
axs[1][1].imshow(img2_pred_all[:,:,3],cmap=cmap)
axs[1][2].imshow(img2_pred_all[:,:,4],cmap=cmap)
axs[0][0].set_title('Satellite image in 2019')
axs[0][1].set_title(modelnamesfull[0]+' (%.2f'%(acclist[0])+')') 
axs[0][2].set_title(modelnamesfull[1]+' (%.2f'%(acclist[1])+')') 
axs[1][0].set_title(modelnamesfull[2]+' (%.2f'%(acclist[2])+')') 
axs[1][1].set_title(modelnamesfull[3]+' (%.2f'%(acclist[3])+')') 
axs[1][2].set_title(modelnamesfull[4]+' (%.2f'%(acclist[4])+')')
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_comparison_2019.jpeg', bbox_inches='tight')
plt.show()

#Integrate classification maps of different models through voting 
img2_pred_vote = img2_pred
for i in range (0,m):
    for j in range(0,n):
        nums = img2_pred_all[i,j,:]
        img2_pred_vote[i][j] = np.argmax(np.bincount(nums)) #mode value

#Compute accuracy for ensemble learning
values = dfsample2[['Row','Column']].values
m,n = dfsample2[['Row','Column']].shape
sample_pred = []
for i in range(0,m):
    row = values[i][0] #row
    col = values[i][1] #column
    label_pred = img2_pred_vote[row][col]
    sample_pred.append(label_pred)
dfsample2['ensemble_pred'] = sample_pred
dfsample2test = dfsample2.clip(y_test, axis=0).dropna()
y_test = dfsample2test['Label']
y_pred = dfsample2test['ensemble_pred']
accuracy = accuracy_score(y_test, y_pred)
print ("Ensemble learning")
print ("Accuracy: %.2f"%(accuracy))

#Plot ensemble learning results
sns.set(font_scale=1, style="ticks", palette="bright", color_codes=True)
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7), dpi=300)
cmap = colors.ListedColormap(['darkgoldenrod','darkgrey','white','green','brown'])
ax1.imshow(img2rgb)
ax2.imshow(img2_pred_vote,cmap=cmap)
ax1.set_title('Satellite image in 2019')
ax2.set_title('Ensemble learning (%.2f'%(accuracy)+')') 
plt.subplots_adjust(wspace=0.1, hspace=0)
#plt.savefig(outdir+'classfication_maps_ensemble_2019.jpeg', bbox_inches='tight')
plt.show()
