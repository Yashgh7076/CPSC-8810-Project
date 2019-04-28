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
import json

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
    max_value = 1
    min_value = 0
    # Normalize images back to range [0,1]
    for i in range(N):
        i1 = images[i, :, :, 0]
        i1_std = (i1 - np.amin(i1, axis = (0,1)))/ (np.amax(i1, axis = (0,1)) - np.amin(i1, axis = (0,1)))
        i1 = i1_std*(max_value - min_value) + min_value
        
        i2 = images[i, :, :, 1]
        i2_std = (i2 - np.amin(i2, axis = (0,1)))/ (np.amax(i2, axis = (0,1)) - np.amin(i2, axis = (0,1)))
        i2 = i2_std*(max_value - min_value) + min_value
        
        i3 = images[i, :, :, 2]
        i3_std = (i3 - np.amin(i3, axis = (0,1)))/ (np.amax(i3, axis = (0,1)) - np.amin(i3, axis = (0,1)))
        i3 = i3_std*(max_value - min_value) + min_value
        
        images[i, :, :, 0] = i1
        images[i, :, :, 1] = i2
        images[i, :, :, 2] = i3
    
    return(images)
    
def jitter(dataset, brightness):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))    
    bright = np.ones((ROWS, COLS, 3)) + np.random.uniform(-brightness, brightness, (ROWS, COLS, 3))    
    images = dataset * bright
    
    max_value = 1
    min_value = 0
    # Normalize images back to range [0,1]
    for i in range(N):
        i1 = images[i, :, :, 0]
        i1_std = (i1 - np.amin(i1, axis = (0,1)))/ (np.amax(i1, axis = (0,1)) - np.amin(i1, axis = (0,1)))
        i1 = i1_std*(max_value - min_value) + min_value
        
        i2 = images[i, :, :, 1]
        i2_std = (i2 - np.amin(i2, axis = (0,1)))/ (np.amax(i2, axis = (0,1)) - np.amin(i2, axis = (0,1)))
        i2 = i2_std*(max_value - min_value) + min_value
        
        i3 = images[i, :, :, 2]
        i3_std = (i3 - np.amin(i3, axis = (0,1)))/ (np.amax(i3, axis = (0,1)) - np.amin(i3, axis = (0,1)))
        i3 = i3_std*(max_value - min_value) + min_value
        
        images[i, :, :, 0] = i1
        images[i, :, :, 1] = i2
        images[i, :, :, 2] = i3
    
    return(images)

def contrast(dataset, contrast):
    N = dataset.shape[0]
    
    images = np.zeros((N, ROWS, COLS, 3))
    coef = np.array([[[0.299, 0.587, 0.114]]])
    max_value = 1
    min_value = 0
        
    for i in range(N):
        alpha = np.ones((ROWS, COLS, 3)) + np.random.uniform(-contrast, contrast, (ROWS, COLS, 3))
        img = np.zeros((ROWS, COLS, 3))
        gray = np.zeros((ROWS, COLS, 3))
        
        img = dataset[i, :, :, :]
            
        gray = img * coef
            
        gray = (3.0 * (1.0 - alpha) / (gray.size)) * np.sum(gray)
            
        img = img*alpha
            
        img= img + gray
            
                
        i1 = img[:, :, 0]
        i1_std = (i1 - np.amin(i1, axis = (0,1)))/ (np.amax(i1, axis = (0,1)) - np.amin(i1, axis = (0,1)))
        i1 = i1_std*(max_value - min_value) + min_value
        
        i2 = img[:, :, 1]
        i2_std = (i2 - np.amin(i2, axis = (0,1)))/ (np.amax(i2, axis = (0,1)) - np.amin(i2, axis = (0,1)))
        i2 = i2_std*(max_value - min_value) + min_value
        
        i3 = img[:, :, 2]
        i3_std = (i3 - np.amin(i3, axis = (0,1)))/ (np.amax(i3, axis = (0,1)) - np.amin(i3, axis = (0,1)))
        i3 = i3_std*(max_value - min_value) + min_value
        
        images[i, :, :, 0] = i1
        images[i, :, :, 1] = i2
        images[i, :, :, 2] = i3
            
        
    return(images)
   
with open ('DL_labels.json') as json_file:
    data = json.load(json_file)

filename_1 = 'Labelling/'
a = os.listdir(filename_1)

m = len(a)
    
dataset = np.zeros(shape = (m, ROWS, COLS, 3))

dataset = read_from_folder(filename_1, 0, 1)

j = 0
num = len(data)

images = np.zeros(shape = (100, 75, 75, 3))
labels = np.zeros(shape = (100))

for i in range(5):
    if 'Bully' in data[i]['Label']:
        list_c = data[i]['Label']['Bully']
        j = len(list_c) # No. of bullies
        for k in range(j):
            x1 = list_c[k]['geometry'][0]['x']
            y1 = list_c[k]['geometry'][0]['y']
            x2 = list_c[k]['geometry'][1]['x']
            y2 = list_c[k]['geometry'][1]['y']
            x3 = list_c[k]['geometry'][2]['x']
            y3 = list_c[k]['geometry'][2]['y']
            x4 = list_c[k]['geometry'][3]['x']
            y4 = list_c[k]['geometry'][3]['y']
            xmin = min(x1,x2,x3,x4)
            xmax = max(x1,x2,x3,x4)
            ymin = min(y1,y2,y3,y4)
            ymax = max(y1,y2,y3,y4)

            rows = ymax - ymin
            cols = xmax - xmin
            temp = np.zeros((rows, cols, 3))

            temp = dataset[i, ymin:ymax, xmin:xmax, :]
            temp = np.resize(temp, (75, 75, 3))            

            #np.append(images, temp, axis = 0)
            #np.append(labels, 1, axis = 0)
            images[i, :, :, :] = temp
            labels[i] = 1


    if 'Victim' in data[i]['Label']:
        list_c = data[i]['Label']['Victim']
        j = len(list_c) # No. of victims
        for k in range(j):            
            x5 = list_c[k]['geometry'][0]['x']
            y5 = list_c[k]['geometry'][0]['y']
            x6 = list_c[k]['geometry'][1]['x']
            y6 = list_c[k]['geometry'][1]['y']
            x7 = list_c[k]['geometry'][2]['x']
            y7 = list_c[k]['geometry'][2]['y']
            x8 = list_c[k]['geometry'][3]['x']
            y8 = list_c[k]['geometry'][3]['y']
            xmin = min(x5,x6,x7,x8)
            xmax = max(x5,x6,x7,x8)
            ymin = min(y5,y6,y7,y8)
            ymax = max(y5,y6,y7,y8)

            rows = ymax - ymin
            cols = xmax - xmin
            temp = np.zeros((rows, cols, 3))

            temp = dataset[i, ymin:ymax, xmin:xmax, :]
            temp = np.resize(temp, (75, 75, 3))

            #np.append(images, temp, axis = 0)
            #np.append(labels, 0, axis = 0)
            images[i, :, :, :] = temp
            labels[i] = 0
            
plt.imshow(dataset[5, :, :, :])



