# Import libraries
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import math
import cv2 as cv
import json

# Set global values
ROWS = 224
COLS = 224
newsize = 105
window = 9
tf.set_random_seed(0)

#-----------Functions start here-----------#
def min_max_normalize(image, min_value, max_value): 
    
    for c in range(3):
        img = image[:, :, c]
        img_std = (img - np.amin(img, axis = (0,1)))/ (np.amax(img, axis = (0,1)) - np.amin(img, axis = (0,1)))
        img_scl = img_std*(max_value - min_value) + min_value
        image[:, :, c] = img_scl    

def skip_images(skip_list):

    number = len(skip_list)

    return(number)

def read_from_folder(filename, data, D, labels):
    a = os.listdir(filename) # Sorted list of files
    m = 1840
    
    j = 0
    
    l = 0    
    
    for i in range(m):
        print(i)
        if(i in [878, 899, 965, 1069, 1071, 1078, 1086, 1094, 1095 , 1096, 1097, 1098, 1104, 1133, 1144, 1165, 1184, 1201, 1210, 1221, 1222, 1225, 1229, 1252, 1286, 1327, 1329, 1334, 1361,1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407 ,1414, 1422, 1424, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440 , 1441, 1442, 1443 ,1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1495, 1514, 1524, 1546, 1560, 1562, 1566, 1567, 1579, 1580, 1581, 1590, 1592, 1593, 1596, 1599, 1623, 1628, 1641, 1649, 1652, 1655, 1660, 1686, 1724, 1725, 1733, 1742, 1750, 1756, 1759, 1762, 1773, 1783, 1786, 1800, 1801, 1811, 1817, 1818, 1831, 1836]):
            continue
        
        b = a[i]
        c = os.path.join(filename, b) # Full path to read image        
        # Read image
        img = Image.open(c)
        #img = img.resize((COLS, ROWS), Image.ANTIALIAS)
        img = np.array(img, dtype = np.float64)  
        
        # Detect if image is grayscale
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img    
            temp = temp/255.0

            # Copy grayscale into each channel seperately
            img = np.zeros((temp.shape[0], temp.shape[1], 3))
            img[:, :, 0] = temp
            img[:, :, 1] = temp
            img[:, :, 2] = temp  
        elif(len(img.shape) == 3):
            i1 = img[:, :, 0]        
            i1 = i1/255.0

            i2 = img[:, :, 1]        
            i2 = i2/255.0

            i3 = img[:, :, 2]            
            i3 = i3/255.0

            img[:, :, 0] = i1
            img[:, :, 1] = i2
            img[:, :, 2] = i3    
              
        # Part 2 of code
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
                xmin_b = min(x1,x2,x3,x4)
                xmax_b = max(x1,x2,x3,x4)
                ymin_b = min(y1,y2,y3,y4)
                ymax_b = max(y1,y2,y3,y4)
                cols_b = xmax_b - xmin_b
                rows_b = ymax_b - ymin_b
                #print('Bully: xmin, xmax, ymin, ymax \n',(xmin_b, xmax_b, ymin_b, ymax_b))
                temp = np.zeros((rows_b, cols_b, 3))

                temp = img[ymin_b:ymax_b, xmin_b:xmax_b, :]
                              
                temp = cv.resize(temp, (newsize, newsize), interpolation =  cv.INTER_LINEAR)  
                
                D[l, :, :, :] = temp
                min_max_normalize(D[l, :, :, :], 0, 1)                
                
                labels[l] = 1
                l = l + 1
                

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
                xmin_v = min(x5,x6,x7,x8)
                xmax_v = max(x5,x6,x7,x8)
                ymin_v = min(y5,y6,y7,y8)
                ymax_v = max(y5,y6,y7,y8)
                #print('Victim: xmin, xmax, ymin, ymax \n',(xmin_v, xmax_v, ymin_v, ymax_v))

                cols_v = xmax_v - xmin_v
                rows_v = ymax_v - ymin_v
                temp = np.zeros((rows_v, cols_v, 3))

                temp = img[ymin_v:ymax_v, xmin_v:xmax_v, :]
                                
                temp = cv.resize(temp, (newsize, newsize), interpolation =  cv.INTER_LINEAR)
                
                D[l, :, :, :] = temp
                min_max_normalize(D[l, :, :, :], 0, 1) 
                
                labels[l] = 0 # Victim => 0 
                l = l + 1
                

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

def avg_smooth(dataset, region):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, ROWS, COLS, 3))
    img = np.zeros(shape = (ROWS, COLS))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            images[i, :, :, c] = cv.blur(img,(region,region))
            
    return(images)

# Read files and labels
filename_1 = 'Labelling/'
N = 1840

with open ('DL_labels.json') as json_file:
    data = json.load(json_file)

miss_list = [878, 899, 965, 1069, 1071, 1078, 1086, 1094, 1095 , 1096, 1097, 1098, 1104, 1133, 1144, 1165, 1184, 1201, 1210, 1221, 1222, 1225, 1229, 1252, 1286, 1327, 1329, 1334, 1361,1400, 1401, 1402, 1403, 1404, 1405, 1406, 1407 ,1414, 1422, 1424, 1430, 1431, 1432, 1433, 1434, 1435, 1436, 1437, 1438, 1439, 1440 , 1441, 1442, 1443 ,1444, 1445, 1446, 1447, 1448, 1449, 1450, 1451, 1452, 1453, 1453, 1454, 1455, 1456, 1457, 1458, 1459, 1460, 1495, 1514, 1524, 1546, 1560, 1562, 1566, 1567, 1579, 1580, 1581, 1590, 1592, 1593, 1596, 1599, 1623, 1628, 1641, 1649, 1652, 1655, 1660, 1686, 1724, 1725, 1733, 1742, 1750, 1756, 1759, 1762, 1773, 1783, 1786, 1800, 1801, 1811, 1817, 1818, 1831, 1836]

skip_these = skip_images(miss_list)
L10 = N - skip_these

D = np.zeros(shape = (L10, newsize, newsize, 3))
labels = np.zeros(shape = (L10))

D_temp = np.zeros(shape = (L10, newsize, newsize, 3))
labels_temp = np.zeros(L10)

read_from_folder(filename_1, data, D, labels)  

D_temp = D # Copy images + labels
labels_temp = labels

del D # Free images + labels temporarily
del labels

total_number = 10*L10

D = np.zeros(shape = (total_number, newsize, newsize, 3))
labels = np.zeros(shape = (total_number))

D[0:L10,:,:,:] = D_temp
labels[0:L10] = labels_temp

del D_temp
del labels_temp

# Data augmentation part
D[L10:(2*L10), :, :, :] = flip_images(D[0:L10, :, :, :]) # Flip images horizontally
D[(2*L10):(3*L10),:,:,:] = jitter(D[0:L10, :, :, :], 0.25) # Brightness jitter

D[(3*L10):(4*L10),:,:,:] = flip_images(D[0:L10, :, :, :]) # Flip + Additive Gaussian noise
D[(3*L10):(4*L10),:,:,:] = add_noise(D[(3*L10):(4*L10), :, :, :], 0, 0.25)

D[(4*L10):(5*L10),:,:,:] = translate(D[0:L10, :, :, :], 25, 25) # Geometric transforms
D[(5*L10):(6*L10),:,:,:] = rotate_90(D[0:L10, :, :, :])
D[(6*L10):(7*L10),:,:,:] = affine_transform(D[0:L10, :, :, :])
D[(7*L10):(8*L10),:,:,:] = perspective_transform(D[0:L10, :, :, :])

D[(8*L10):(9*L10),:,:,:] = add_noise(D[0:L10, :, :, :], 0, 0.25) # Additive Gaussian noise + Smoothing -> Uneven blur
D[(8*L10):(9*L10),:,:,:] = avg_smooth(D[(8*L10):(9*L10),:,:,:], 5)

D[(9*L10):(10*L10),:,:,:] = rotate_90(D[0:L10, :, :, :]) # Rotate + Additive Gaussian noise
D[(9*L10):(10*L10),:,:,:] = add_noise(D[(9*L10):(10*L10), :, :, :], 0, 0.25)

# Augment labels
augment_labels(labels, L10, L) # augment labels for flipped images

augment_labels(labels, (2*L10), L) # augment labels for jitter images

augment_labels(labels, (3*L10), L) # augment labels for flipped + noise images

augment_labels(labels, (4*L10), L)

augment_labels(labels, (5*L10), L)

augment_labels(labels, (6*L10), L)

augment_labels(labels, (7*L10), L)

augment_labels(labels, (8*L10), L)

augment_labels(labels, (9*L10), L)


