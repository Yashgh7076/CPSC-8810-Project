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

# Libraries for training and plotting
import tensorflow as tf


# Define Output Image Size Here
ROWS = 64 #240
COLS = 96 #360

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
            # Normalize the image
            temp = img;    
           
            temp = (temp - np.mean(temp, axis = (0,1)))/np.std(temp, axis = (0,1))
            
            temp = resize(temp, output_shape = (ROWS, COLS))
              
            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        
        i2 = img[:, :, 1]
        i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        
        i3 = img[:, :, 2]
        i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        
        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        img = resize(img, output_shape = (ROWS, COLS, 3))
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
        img = io.imread(d)
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;
            
            temp = (temp - np.mean(temp, axis = (0,1)))/ np.std(temp, axis = (0,1))
            
            temp = resize(temp, output_shape = (ROWS, COLS))
              
            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        
        i2 = img[:, :, 1]
        i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        
        i3 = img[:, :, 2]
        i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        
        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        img = resize(img, output_shape = (ROWS, COLS, 3))
        images[i, :, :, :] = img
        
    return(images)
        
def get_next_batch(index, dataset, classes, batch_size):
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.uint8)
    
    return(X_batch,Y_batch) 
        
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
number = L9 # Number of non - bullying images needed
D2 = np.zeros(shape = (number, ROWS, COLS, 3))
D2 = read_nonbullying(filename_10, number)

# Full Dataset
D = np.zeros(shape = (L9 + number, ROWS, COLS, 3))
D[0:L9,:,:,:] = D1
D[L9:L9 + number,:,:,:] = D2

del D1
del D2

# Create labels Bullying => 1 // Non - bullying => 0
labels = np.zeros(shape = (L9 + number), dtype = np.uint8)
labels[0:L9] = 1

# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
Y = tf.placeholder(tf.uint8,[None])
depth = 2 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
print(Y_onehot.shape)
#lr = tf.placeholder(tf.float32)
#pkeep = tf.placeholder(tf.float32)
pkeep = 0.5

# Specify parameters of the NN model & training 
depth_1 = 6 # CNN Layer 1 o/p channels
depth_2 = 4
depth_3 = 2
fc = 45
batch_size = 10


# Create weights & biases
# Layer 1
W1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 3, depth_1], stddev = 1.0))
B1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_1]))
# Layer 2
W2 = tf.Variable(tf.truncated_normal(shape = [4, 4, depth_1, depth_2], stddev = 1.0))
B2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_2]))
# Layer 3
#W3 = tf.Variable(tf.truncated_normal(shape = [3, 3, depth_2, depth_3], stddev = 1.0))
#B3 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_3]))

# Fully Connected Layer
W4 = tf.Variable(tf.truncated_normal(shape = [24 * 16 * depth_2, fc], stddev = 1.0))
B4 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc]))
# Output Layer
W5 = tf.Variable(tf.truncated_normal(shape = [fc, 2], stddev = 1.0))
B5 = tf.Variable(tf.constant(0.1, tf.float32, shape = [2]))

# Create model
# CNN => 2 Convolutional Layers // 1 Fully Connected Layer
Y1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME') + B1)
Y1_max = tf.nn.max_pool(Y1, ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')

Y2 = tf.nn.relu(tf.nn.conv2d(Y1_max, W2, strides = [1,1,1,1], padding = 'SAME') + B2)
Y2_max = tf.nn.max_pool(Y2, ksize = [1,2,2,1], strides = [1,1,1,1], padding = 'SAME')

YY = tf.reshape(Y2_max, shape = [-1, 24 * 16 * depth_2])
Y4 = tf.nn.relu(tf.matmul(YY, W4) + B4)

Y4_drop = tf.nn.dropout(Y4, keep_prob = pkeep)

Y_logits= tf.matmul(Y4, W5) + B5

Y_pred = tf.nn.softmax(Y_logits)

# Evaluate model loss
cross_entropy = tf.losses.softmax_cross_entropy(Y_onehot, Y_logits)
cross_entropy = tf.reduce_mean(cross_entropy)

#Calculate accuracy for each mini batch
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Define optmizer
optimizer = tf.train.AdamOptimizer(0.001)
train_step = optimizer.minimize(cross_entropy)

# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(10):
    batch = 1
    train_loss = 0
    train_acc = 0

    test_loss = 0
    test_acc = 0
    
    for i in range(0, L9 + number, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, D, labels, batch_size)
        sess.run(train_step,feed_dict = {X: batch_X, Y: batch_Y})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = D)
        train_acc = train_acc + a
        train_loss = train_loss + c
    
        test_data = {X: D, Y:labels}
        a,c = sess.run([accuracy,cross_entropy], feed_dict = test_data)
        test_acc = test_acc + a
        test_loss = test_loss + c
        batch = batch + 1
        
    train_loss = train_loss/batch
    train_acc = train_acc/batch
    
    test_loss = test_loss/batch
    test_acc = test_acc/batch
    
    print("Epoch",epoch + 1,"Train Loss",train_loss,"Train Acc",train_acc)
    print("Epoch",epoch + 1,"Test Loss",test_loss,"Test Acc",test_acc)
    print("\n")

