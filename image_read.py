# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:54:40 2019

@author: Yadnyesh
"""

import os
import skimage.io as io
from skimage.transform import resize
import numpy as np
import h5py

# Define Output Image Size Here
ROWS = 240
COLS = 360

# Read Folder_1
filename = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\Folder_1'

a = sorted(os.listdir(filename)) # Collects names of all images in directory

m = len(a)

# Read Folder_2
# filename_2 = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\Folder_2'

# p = sorted(os.listdir(filename_2)) # Collects names of all images in directory

# n = len(p)

images = np.zeros(shape = (m, ROWS, COLS, 3))


# Read images from Folder_1
for i in range(m):
    b = a[i]
    c = os.path.join(filename, b) # Full path to read image
    #print(c)
    
    # Read image
    img = io.imread(c)
    
    # Detect if the image is grayscale
    if(len(img.shape) == 2):
        temp = resize(img, output_shape = (ROWS, COLS))
        # Copy grayscale into each channel seperately
        images[i, :, :, 0] = temp
        images[i, :, :, 1] = temp
        images[i, :, :, 2] = temp        
        continue
    
    
    img = resize(img, output_shape = (ROWS, COLS, 3))
    images[i, :, :, :] = img
    

# Read images from Folder_2
#for i in range(n):
#    b = p[i]
#    c = os.path.join(filename_2, b) # Full path to read image
    #print(c)
    
    # Read image
#    img = io.imread(c)
    
    # Detect if the image is grayscale
#    if(len(img.shape) == 2):
#        temp = resize(img, output_shape = (ROWS, COLS))
#        # Copy grayscale into each channel seperately
#        images[i, :, :, 0] = temp
#        images[i, :, :, 1] = temp
#        images[i, :, :, 2] = temp        
#        continue
    
    
#    img = resize(img, output_shape = (ROWS, COLS, 3))
#    images[(m + i), :, :, :] = img    
    
# Save array to disk
#np.save(file = 'F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\dataset_1.npy', arr = images)
h5f = h5py.File("F:\Clemson University\ECE 8810_Deep Learning\Project\gossiping\data.hdf5", "w")
h5f.create_dataset('dataset_1j', data=images, dtype = np.float64)  

  