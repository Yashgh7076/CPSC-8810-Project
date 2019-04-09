# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 17:54:15 2019

@author: Yadnyesh
"""

# Import libraries
import numpy as np
#import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
#import math

# Set global values
ROWS = 224
COLS = 224

# Functions go here
def read_from_folder(filename, max_value, min_value):
    a = sorted(os.listdir(filename)) # Sorted list of files
    
    m = len(a)
    
    images = np.zeros(shape = (m, ROWS, COLS, 3))
    
    for i in range(m):
        b = a[i]
        c = os.path.join(filename, b) # Full path to read image
        print(c)
        # Read image
        img = Image.open(c)
        img = img.resize((COLS, ROWS), Image.ANTIALIAS)
        img = np.array(img, dtype = np.float64)
        
        # Detect if image is grayscale
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;    
           
            temp_std = (temp - np.amin(temp, axis = (0,1)))/ (np.amax(temp, axis = (0,1)) - np.amin(temp, axis = (0,1)))
            temp = temp_std*(max_value - min_value) + min_value
            #temp = (temp - np.mean(temp, axis = (0,1)))/np.std(temp, axis = (0,1))                    
            #temp = temp/255.0

            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1_std = (i1 - np.amin(i1, axis = (0,1)))/ (np.amax(i1, axis = (0,1)) - np.amin(i1, axis = (0,1)))
        i1 = i1_std*(max_value - min_value) + min_value
        #i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        #i1 = i1/255.0

        i2 = img[:, :, 1]
        i2_std = (i2 - np.amin(i2, axis = (0,1)))/ (np.amax(i2, axis = (0,1)) - np.amin(i2, axis = (0,1)))
        i2 = i2_std*(max_value - min_value) + min_value
        #i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        #i2 = i2/255.0

        i3 = img[:, :, 2]
        i3_std = (i3 - np.amin(i3, axis = (0,1)))/ (np.amax(i3, axis = (0,1)) - np.amin(i3, axis = (0,1)))
        i3 = i3_std*(max_value - min_value) + min_value
        #i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        #i3 = i3/255.0

        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        images[i, :, :, :] = img
        
        
    return(images)


def read_nonbullying(filename, number, max_value, min_value):
    a = sorted(os.listdir(filename))
    
    m = len(a)
    
    n = np.random.randint(0,m, size = (number))
    
    b = len(n)
    
    images = np.zeros(shape = (b, ROWS, COLS, 3))
    
    for i in range(b):
        c = a[n[i]]
        d = os.path.join(filename, c)
        print(d)
        
        # Read image
        img = Image.open(d)
        img = img.resize((COLS, ROWS))
        img = np.array(img, dtype = np.float64)
        
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;
            
            temp_std = (temp - np.amin(temp, axis = (0,1)))/ (np.amax(temp, axis = (0,1)) - np.amin(temp, axis = (0,1)))
            temp = temp_std*(max_value - min_value) + min_value
            #temp = temp/255

            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1_std = (i1 - np.amin(i1, axis = (0,1)))/ (np.amax(i1, axis = (0,1)) - np.amin(i1, axis = (0,1)))
        i1 = i1_std*(max_value - min_value) + min_value
        #i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        #i1 = i1/255

        i2 = img[:, :, 1]
        i2_std = (i2 - np.amin(i2, axis = (0,1)))/ (np.amax(i2, axis = (0,1)) - np.amin(i2, axis = (0,1)))
        i2 = i2_std*(max_value - min_value) + min_value
        #i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        #i2 = i2/255

        i3 = img[:, :, 2]
        i3_std = (i3 - np.amin(i3, axis = (0,1)))/ (np.amax(i3, axis = (0,1)) - np.amin(i3, axis = (0,1)))
        i3 = i3_std*(max_value - min_value) + min_value
        #i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        #i3 = i3/255

        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        images[i, :, :, :] = img
        
    return(images)

def flip_images(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    
    for i in range(N):
        for j in range(3):
            temp = dataset[i,:,:,j]
            rot  = np.fliplr(temp) # Actual Flip LR operation
            images[i,:,:,j] = rot
            
    return(images)
    
def add_noise(dataset, mean, std_dev):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    
    noise = np.random.normal(mean, std_dev, (ROWS, COLS, 3))
    
    images = dataset + noise
    
    return(images)
    
def translate(dataset, topleft, bottomleft, topright, bottomright):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    
    
    # Sequence of shift => Left / Up / Right / Down
    # down = topleft // up = bottomleft // left = topright // right = bottomright
    for i in range(N):
        img = np.zeros((ROWS, COLS)) 
        #temp1 = np.zeros((ROWS, COLS)) 
        #temp2 = np.zeros((ROWS, COLS)) 
        #temp3 = np.zeros((ROWS, COLS))
        temp4 = np.zeros((ROWS, COLS))            
        for j in range(3):             
            img = dataset[i, :, :, j]     
            #temp1[:,left:(COLS -1)] = img[:,0:(COLS - 1) - left]
            #temp2[0:(ROWS - 1) - up, left:(COLS - 1)] = temp1[up:ROWS -1, left:(COLS - 1)]
            #temp3[0:(ROWS - 1) - up, left:(COLS - 1) - right] = img[0:(ROWS -1) - up, 0:(COLS - 1) - right - left]
            temp4[topleft:(ROWS - 1) - bottomleft, topright:(COLS -1) - bottomright] = img[topleft:(ROWS - 1) - bottomleft, topright:(COLS - 1) - bottomright]
            images[i, :, :, j] = temp4 
    
    
    return(images)
   
# Read Folder_1
filename_1 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\Folder_1'
a1 = sorted(os.listdir(filename_1))
l1 = len(a1)

D = np.zeros(shape = (l1, ROWS, COLS, 3))
D_flip = np.zeros(shape = (l1, ROWS, COLS, 3))

D = read_from_folder(filename_1, 1, 0)

D_flip = flip_images(D)

D_noise = add_noise(D, 0, 0.25)

D_translate = translate(D, 25, 25, 25, 25)

#plt.figure(1)
#plt.imshow(D[0,:,:,:])

#plt.figure(2)
#plt.imshow(D_flip[0,:,:,:])

#plt.figure(3)
#plt.imshow(D_noise[0,:,:,:])

plt.figure(4)
plt.imshow(D[6,:,:,:])

#plt.figure(5)
#plt.imshow(D[6,:,:,:])
    
#plt.figure(6)
#plt.imshow(D_flip[6,:,:,:])

#plt.figure(7)
#plt.imshow(D_noise[6,:,:,:])

plt.figure(8)
plt.imshow(D_translate[6,:,:,:])
