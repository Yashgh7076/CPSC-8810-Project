# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:38:41 2019

@author: Yadnyesh
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:24:14 2019

@author: Yadnyesh
"""

# Import libraries
import numpy as np
import os
from PIL import Image
import cv2 as cv
import matplotlib.pyplot as plt

# Set global values
ROWS = 224
COLS = 224
newsize = 105

#-----------Functions start here-----------#
def read_from_folder(filename):
    a = sorted(os.listdir(filename)) # Sorted list of files
    
    m = len(a)
    
    images = np.zeros(shape = (m, ROWS, COLS, 3))
    
    for i in range(m):
        print(i)        
        b = a[i]
        c = os.path.join(filename, b) # Full path to read image
        print(c)
        
        img = Image.open(c)
        img = img.resize((COLS, ROWS), Image.ANTIALIAS)
        img = np.array(img, dtype = np.float64)
        
        # Detect if image is grayscale
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;    
            temp = temp/255.0
            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = i1/255.0

        i2 = img[:, :, 1]
        i2 = i2/255.0

        i3 = img[:, :, 2]
        i3 = i3/255.0

        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        images[i, :, :, :] = img
        
        
    return(images)


def read_nonbullying(filename, number):
    a = sorted(os.listdir(filename))
    
    m = len(a)
    
    n = np.random.randint(0,m, size = (number))
    
    b = len(n)
    
    images = np.zeros(shape = (b, ROWS, COLS, 3))
    
    for i in range(b):
        c = a[n[i]]
        d = os.path.join(filename, c)
               
        # Read image
        img = Image.open(d)
        img = img.resize((COLS, ROWS))
        img = np.array(img, dtype = np.float64)
        
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;
            temp = temp/255.0
            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = i1/255.0

        i2 = img[:, :, 1]
        i2 = i2/255.0

        i3 = img[:, :, 2]
        i3 = i3/255.0

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
    
def jitter(dataset, brightness):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))    
    bright = np.ones((ROWS, COLS, 3)) + np.random.uniform(-brightness, brightness, (ROWS, COLS, 3))    
    images = dataset * bright
    
    return(images)

def get_next_batch(index, dataset, classes, batch_size):
    
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.int32)
    
    return(X_batch,Y_batch) 

def augment_labels(labels, previous, L):
    
    labels[previous: previous + L[0]] = 1 # Do not use l10 -> total number of stanford images
    labels[previous + L[0]: previous + L[1]] = 2
    labels[previous + L[1]: previous + L[2]] = 3
    labels[previous + L[2]: previous + L[3]] = 4
    labels[previous + L[3]: previous + L[4]] = 5
    labels[previous + L[4]: previous + L[5]] = 6
    labels[previous + L[5]: previous + L[6]] = 7
    labels[previous + L[6]: previous + L[7]] = 8
    labels[previous + L[7]: previous + L[8]] = 9
    labels[previous + L[8]: previous + L[9]] = 0

def min_max_normalize(dataset, min_value, max_value):

    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3)) 
    img = np.zeros(shape = (ROWS, COLS))

    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            img_std = (img - np.amin(img, axis = (0,1)))/ (np.amax(img, axis = (0,1)) - np.amin(img, axis = (0,1)))
            img_scl = img_std*(max_value - min_value) + min_value
            images[i, :, :, c] = img_scl
    
    return(images)


def standard_scaler(dataset):

    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3)) 
    img = np.zeros(shape = (ROWS, COLS))

    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            img_scl = (img - np.mean(img, axis = (0, 1)))/np.std(img, axis = (0, 1))
            images[i, :, :, c] = img_scl
    
    return(images)

def translate(dataset, shift_cols, shift_rows):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3)) 
    img = np.zeros(shape = (ROWS, COLS))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            M = np.float32([[1,0,shift_cols],[0,1,shift_rows]])
            images[i,:,:,c] = cv.warpAffine(img,M,(COLS,ROWS))
    
    return(images)
    
def rotate_90(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            M = cv.getRotationMatrix2D(((COLS - 1)/2.0,(ROWS - 1)/2.0),90,1)
            images[i,:,:,c] = cv.warpAffine(img,M,(COLS,ROWS))
    
    return(images)


def affine_transform(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    # 3 points in original image -> rotate 3 points in destination image
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            pts1 = np.float32([[10,10],[200,10],[10,200]])
            pts2 = np.float32([[10,100],[200,10],[100,200]])
            M = cv.getAffineTransform(pts1,pts2)
            images[i, :, :, c] = cv.warpAffine(img,M,(COLS,ROWS))
            
    return(images)
           
def perspective_transform(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    
    # 4 points of perspective transform
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            pts1 = np.float32([[30,30],[150,30],[30,190],[190,190]])
            pts2 = np.float32([[0,0],[ROWS,0],[0,COLS],[ROWS,COLS]])
            M = cv.getPerspectiveTransform(pts1,pts2)
            images[i, :, :, c] = cv.warpPerspective(img,M,(ROWS,COLS))
    
    return(images)
  
def median_blur(dataset, region):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            images[i, :, :, c] = cv.medianBlur(img, region)
            
    return(images)

def avg_smooth(dataset, region):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            images[i, :, :, c] = cv.blur(img,(region,region))
            
    return(images)
    
filename = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\Folder_1'

a1 = sorted(os.listdir(filename))
l1 = len(a1)

show_img = 6
D = np.zeros((l1, ROWS, COLS, 3))
D_trx = D_rot = D_aff = D_per = D_avg = np.zeros((l1, ROWS, COLS, 3))
D = read_from_folder(filename)
plt.figure(1)
plt.imshow(D[show_img, :, :, :])

D_trx = translate(D, 25, 25)
plt.figure(2)
plt.imshow(D_trx[show_img, :, :, :])

D_rot = rotate_90(D)
D_rot = add_noise(D_rot, 0, 0.25)
D_rot = avg_smooth(D_rot, 5)
plt.figure(3)
plt.imshow(D_rot[show_img, :, :, :])

D_aff = affine_transform(D)
plt.figure(4)
plt.imshow(D_aff[show_img, :, :, :])

D_per = perspective_transform(D)
plt.figure(5)
plt.imshow(D_per[show_img, :, :, :])

D_avg = add_noise(D, 0, 0.05)
D_avg = avg_smooth(D_avg, 5)
plt.figure(6)
plt.imshow(D_avg[show_img, :, :, :])





