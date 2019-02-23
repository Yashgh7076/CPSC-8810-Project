# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 16:41:08 2019

@author: Yadnyesh
"""

# Libraries for reading images
import os
import skimage.io as io
from skimage.transform import resize
import numpy as np

# Define Output Image Size Here
ROWS = 240
COLS = 360

# Functions go here

def read_from_folder(filename):
    a = sorted(os.listdir(filename)) # Sorted list of files
    
    m = len(a)
    
    images = np.zeros(shape = (m, ROWS, COLS, 3))
    
    for i in range(m):
        b = a[i]
        c = os.path.join(filename, b) # Full path to read image
        # Read image
        img = io.imread(c)
        # Detect if image is grayscale
        if(len(img.shape) == 2):
            temp = resize(img, output_shape = (ROWS, COLS))
            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        img = resize(img, output_shape = (ROWS, COLS, 3))
        images[i, :, :, :] = img
        
        
    return(images)
        
# Read Folder_1
filename_1 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\Folder_1'
a1 = sorted(os.listdir(filename_1))
l1 = len(a1)


# Read Folder_2
filename_2 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\isolation\Folder_1'
a2 = sorted(os.listdir(filename_2))
l2 = len(a2)
L2 = l1 + l2

# Read Folder_3
filename_3 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\laughing\Folder_1'
a3 = sorted(os.listdir(filename_3))
l3 = len(a3)
L3 = l1 + l2 + l3

# Read Folder_4
filename_4 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\pullinghair\Folder_1'
a4 = sorted(os.listdir(filename_4))
l4 = len(a4)
L4 = l1 + l2 + l3 + l4

# Read Folder_5
filename_5 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\punching\Folder_1'
a5 = sorted(os.listdir(filename_5))
l5 = len(a5)
L5 = l1 + l2 + l3 + l4 + l5

# Read Folder_6
filename_6 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\quarrel\Folder_1'
a6 = sorted(os.listdir(filename_6))
l6 = len(a6)
L6 = l1 + l2 + l3 + l4 + l5 + l6

# Read Folder_7
filename_7 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\slapping\Folder_1'
a7 = sorted(os.listdir(filename_7))
l7 = len(a7)
L7 = l1 + l2 + l3 + l4 + l5 + l6 + l7

# Read Folder_8
filename_8 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\stabbing\Folder_1'
a8 = sorted(os.listdir(filename_8))
l8 = len(a8)
L8 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8

# Read Folder_9
filename_9 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\strangle\Folder_1'
a9 = sorted(os.listdir(filename_9))
l9 = len(a9)
L9 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9

# Read Standford 40 Images
filename_10 = 'F:\Clemson University\ECE 8810_Deep Learning\Stanford40\JPEGImages'
a10 = sorted(os.listdir(filename_10))
l10 = len(a10)

# Bullying images
D1 = np.zeros(shape = (L9, ROWS, COLS, 3))

D1[0:l1, :, :, :] = read_from_folder(filename_1)

D1[l1:L2, :, :, :] = read_from_folder(filename_2)

D1[L2:L3, :, :, :] = read_from_folder(filename_3)

D1[L3:L4, :, :, :] = read_from_folder(filename_4)

D1[L4:L5, :, :, :] = read_from_folder(filename_5)

D1[L5:L6, :, :, :] = read_from_folder(filename_6)

D1[L6:L7, :, :, :] = read_from_folder(filename_7)

D1[L7:L8, :, :, :] = read_from_folder(filename_8)

D1[L8:L9, :, :, :] = read_from_folder(filename_9)

# Non - bullying images
D2 = np.zeros(shape = (l10, ROWS, COLS, 3))
D2 = read_from_folder(filename_10)