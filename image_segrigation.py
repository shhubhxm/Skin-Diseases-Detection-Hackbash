# -*- coding: utf-8 -*-
"""
Created on Fri Feb 26 17:58:25 2021

@author: Divy
"""

#Transfering data to their individual folders
import shutil
import numpy as np
import pandas as pd
from random import random

# Image operations and plotting
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
# %matplotlib inline

# File, path and directory operations
import os
import os.path
import shutil


# Model building
from fastai.vision import *
from fastai.callbacks.hooks import *
import torchvision
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import classification_report
from pathlib import PurePath

# For reproducability
from numpy.random import seed
seed(108)


print(os.listdir("HAM10000"))

# Create a new directory
base = "base"
os.mkdir(base)

#[CREATE FOLDERS INSIDE THE BASE DIRECTORY]

# now we create 7 folders inside 'base':

# train
    # nv
    # mel
    # bkl
    # bcc
    # akiec
    # vasc
    # df
 
# valid
    # nv
    # mel
    # bkl
    # bcc
    # akiec
    # vasc
    # df

# create a path to 'base' to which we will join the names of the new folders
# train
train = os.path.join(base, 'train')
os.mkdir(train)

# valid
valid = os.path.join(base, 'valid')
os.mkdir(valid)


# [CREATE FOLDERS INSIDE THE TRAIN, VALIDATION AND TEST FOLDERS]
# Inside each folder we create seperate folders for each class

# create new folders inside train
nv = os.path.join(train, 'nv')
os.mkdir(nv)
mel = os.path.join(train, 'mel')
os.mkdir(mel)
bkl = os.path.join(train, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(train, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(train, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(train, 'vasc')
os.mkdir(vasc)
df = os.path.join(train, 'df')
os.mkdir(df)





# test
test = os.path.join(base, 'test')
os.mkdir(test)

nv = os.path.join(test, 'nv')
os.mkdir(nv)
mel = os.path.join(test, 'mel')
os.mkdir(mel)
bkl = os.path.join(test, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(test, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(test, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(test, 'vasc')
os.mkdir(vasc)
df = os.path.join(test, 'df')
os.mkdir(df)

# create new folders inside valid
nv = os.path.join(valid, 'nv')
os.mkdir(nv)
mel = os.path.join(valid, 'mel')
os.mkdir(mel)
bkl = os.path.join(valid, 'bkl')
os.mkdir(bkl)
bcc = os.path.join(valid, 'bcc')
os.mkdir(bcc)
akiec = os.path.join(valid, 'akiec')
os.mkdir(akiec)
vasc = os.path.join(valid, 'vasc')
os.mkdir(vasc)
df = os.path.join(valid, 'df')
os.mkdir(df)

import pandas as pd
df = pd.read_csv("HAM10000/HAM10000_metadata.csv")

import numpy as np
from numpy.random import seed
seed(101)
df2=df.iloc[:,1:3]
msk = np.random.rand(len(df2)) < 0.85
train1_df2 = df2[msk]
test_df2 = df2[~msk]
msk1 = np.random.rand(len(train1_df2)) < 0.85
train_df2 = train1_df2[msk1]
validation_df2 = train1_df2[~msk1]

train_df2['dx'].value_counts()

test_df2['dx'].value_counts()

# Set the image_id as the index in df_data
df.set_index('image_id', inplace=True)

# Get a list of images in each of the two folders
folder_1 = os.listdir('HAM10000/HAM10000_images_part_1')
folder_2 = os.listdir('HAM10000/HAM10000_images_part_2')

# Get a list of train , val and test images 
train_df2_list = list(train_df2['image_id'])
validation_df2_list = list(validation_df2['image_id'])
test_df2_list = list(test_df2['image_id'])



# Transfer the train images

for image in train_df2_list:
    
    fname = image + '.jpg'
    label = df.loc[image,'dx']
    
    if fname in folder_1:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(train, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    if fname in folder_2:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(train, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)



# Transfer the val images

for image in validation_df2_list:
    
    fname = image + '.jpg'
    label = df.loc[image,'dx']
    
    if fname in folder_1:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(valid, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    if fname in folder_2:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(valid, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)




for image in test_df2_list:
    
    fname = image + '.jpg'
    label = df.loc[image,'dx']
    
    if fname in folder_1:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_1', fname)
        # destination path to image
        dst = os.path.join(test, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)
    if fname in folder_2:
        # source path to image
        src = os.path.join('HAM10000/HAM10000_images_part_2', fname)
        # destination path to image
        dst = os.path.join(test, label, fname)
        # copy the image from the source to the destination
        shutil.copyfile(src, dst)