# Import libraries
import numpy as np
import tensorflow as tf
import os
from PIL import Image
import matplotlib.pyplot as plt
import math

# Set global values
ROWS = 224
COLS = 224
lr = 0.001
# pkeep = 0.5
window = 5

# Functions go here
def read_from_folder(filename):
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
        img = np.array(img)
        
        # Detect if image is grayscale
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;    
           
            temp = (temp - np.mean(temp, axis = (0,1)))/np.std(temp, axis = (0,1))                    
            # temp = temp/255

            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        # i1 = i1/255

        i2 = img[:, :, 1]
        i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        # i2 = i2/255

        i3 = img[:, :, 2]
        i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        # i3 = i3/255

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
        print(d)
        
        # Read image
        img = Image.open(d)
        img = img.resize((COLS, ROWS), Image.ANTIALIAS)
        img = np.array(img)
        
        if(len(img.shape) == 2):
            # Normalize the image
            temp = img;
            
            temp = (temp - np.mean(temp, axis = (0,1)))/ np.std(temp, axis = (0,1))
            # temp = temp/255

            # Copy grayscale into each channel seperately
            images[i, :, :, 0] = temp
            images[i, :, :, 1] = temp
            images[i, :, :, 2] = temp
            continue
        
        i1 = img[:, :, 0]
        i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))
        # i1 = i1/255

        i2 = img[:, :, 1]
        i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))
        # i2 = i2/255

        i3 = img[:, :, 2]
        i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))
        # i3 = i3/255

        img[:, :, 0] = i1
        img[:, :, 1] = i2
        img[:, :, 2] = i3
        
        images[i, :, :, :] = img
        
    return(images)


def get_next_batch(index, dataset, classes, batch_size):
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.int32)
    
    return(X_batch,Y_batch) 

# Code starts here
# Read Folder_1
filename_1 = 'CPSC_8810/gossiping/'
a1 = sorted(os.listdir(filename_1))
l1 = len(a1)

# Read Folder_2
filename_2 = 'CPSC_8810/isolation/'
a2 = sorted(os.listdir(filename_2))
l2 = len(a2)
L2 = l1 + l2

# Read Folder_3
filename_3 = 'CPSC_8810/laughing/'
a3 = sorted(os.listdir(filename_3))
l3 = len(a3)
L3 = l1 + l2 + l3

# Read Folder_4
filename_4 = 'CPSC_8810/pullinghair/'
a4 = sorted(os.listdir(filename_4))
l4 = len(a4)
L4 = l1 + l2 + l3 + l4

# Read Folder_5
filename_5 = 'CPSC_8810/punching/'
a5 = sorted(os.listdir(filename_5))
l5 = len(a5)
L5 = l1 + l2 + l3 + l4 + l5

# Read Folder_6
filename_6 = 'CPSC_8810/quarrel/'
a6 = sorted(os.listdir(filename_6))
l6 = len(a6)
L6 = l1 + l2 + l3 + l4 + l5 + l6

# Read Folder_7
filename_7 = 'CPSC_8810/slapping/'
a7 = sorted(os.listdir(filename_7))
l7 = len(a7)
L7 = l1 + l2 + l3 + l4 + l5 + l6 + l7

# Read Folder_8
filename_8 = 'CPSC_8810/stabbing/'
a8 = sorted(os.listdir(filename_8))
l8 = len(a8)
L8 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8

# Read Folder_9
filename_9 = 'CPSC_8810/strangle/'
a9 = sorted(os.listdir(filename_9))
l9 = len(a9)
L9 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9

# Read Standford 40 Images
filename_10 = 'CPSC_8810/JPEGImages/'
a10 = sorted(os.listdir(filename_10))
l10 = len(a10)
number = 10

# Initialize the dataset and label container
D = np.zeros(shape = (L9 + number, ROWS, COLS, 3))
labels = np.zeros(shape = (L9 + number))

D[0:l1, :, :, :] = read_from_folder(filename_1)
labels[0:l1] = 1

D[l1:L2, :, :, :] = read_from_folder(filename_2)
labels[l1:L2] = 2

D[L2:L3, :, :, :] = read_from_folder(filename_3)
labels[L2:L3] = 3

D[L3:L4, :, :, :] = read_from_folder(filename_4)
labels[L3:L4] = 4

D[L4:L5, :, :, :] = read_from_folder(filename_5)
labels[L4:L5] = 5

D[L5:L6, :, :, :] = read_from_folder(filename_6)
labels[L5:L6] = 6

D[L6:L7, :, :, :] = read_from_folder(filename_7)
labels[L6:L7] = 7

D[L7:L8, :, :, :] = read_from_folder(filename_8)
labels[L7:L8] = 8

D[L8:L9, :, :, :] = read_from_folder(filename_9)
labels[L8:L9] = 9

D[L9:L9 + number, :, :, :] = read_nonbullying(filename_10, number)
labels[L9:L9 + number] = 0

# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
Y = tf.placeholder(tf.int32,[None])
depth = 10 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
# lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# Specify parameters of the NN model & training 
depth_1 = 10 # CNN Layer 1 o/p channels
depth_2 = 8
depth_3 = 6
fc_1 = 200
fc_2 = 100

# Create weights & biases
# Conv Layer 1
W1 = tf.Variable(tf.truncated_normal(shape = [window, window, 3, depth_1], stddev = 0.1))
B1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_1]))

W1_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, depth_1, depth_2], stddev = 0.1))
B1_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_2]))

W1_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, depth_2, depth_3], stddev = 0.1))
B1_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [depth_3]))

# FC Layer 1
W2 = tf.Variable(tf.truncated_normal(shape = [28 * 28 * depth_3, fc_1], stddev = 0.1))
B2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc_1]))

# FC Layer 2
W3 = tf.Variable(tf.truncated_normal(shape = [fc_1, fc_2], stddev = 0.1))
B3 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc_2]))

# Output Layer
W4 = tf.Variable(tf.truncated_normal(shape = [fc_2, 10], stddev = 0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, shape = [10]))

# Create model
# CNN => 2 Convolutional Layers // 2 Fully Connected Layer
Y1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME') + B1 # Image Size => 224 x 224
Y1_max = tf.nn.max_pool(Y1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y1_out = tf.nn.relu(Y1_max) # Image Size 112 x 112

Y1_1 = tf.nn.conv2d(Y1_out, W1_1, strides = [1,1,1,1], padding = 'SAME') + B1_1 # Image Size => 224 x 224
Y1_1_max = tf.nn.max_pool(Y1_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y1_1_out = tf.nn.relu(Y1_1_max) # Image Size 112 x 112

Y1_2 = tf.nn.conv2d(Y1_1_out, W1_2, strides = [1,1,1,1], padding = 'SAME') + B1_2 # Image Size => 224 x 224
Y1_2_max = tf.nn.max_pool(Y1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y1_2_out = tf.nn.relu(Y1_2_max) # Image Size 112 x 112

YY = tf.reshape(Y1_2_out, shape = [-1, 28 * 28 * depth_3])
Y2 = tf.matmul(YY, W2) + B2
Y2_drop = tf.nn.dropout(Y2, keep_prob = pkeep)
Y2_out = tf.nn.relu(Y2_drop)

Y3 = tf.matmul(Y2_out, W3) + B3
Y3_drop = tf.nn.dropout(Y3, keep_prob = pkeep)
Y3_out = tf.nn.relu(Y3_drop)

Y_logits= tf.matmul(Y3_out, W4) + B4

Y_pred = tf.nn.softmax(Y_logits)

cross_entropy = tf.losses.softmax_cross_entropy(Y_onehot, Y_logits)
cross_entropy = tf.reduce_mean(cross_entropy)

#Calculate accuracy for each mini batch
correct_prediction = tf.equal(tf.argmax(Y_pred, 1), tf.argmax(Y_onehot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Define optmizer
optimizer = tf.train.AdamOptimizer(lr)
train_step = optimizer.minimize(cross_entropy)

# Initialization
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 64 # 32 // Change code to check if code is in train/test

for epoch in range(125):
    batch = 1
    train_loss = 0
    train_acc = 0    
    
    # Randomly shuffle the dataset
    indices = np.arange(L9 + number)
    np.random.shuffle(indices)
    D = D[indices, :, :, :]
    labels = labels[indices]
    
    # Split the data into train & test for each epoch
    index = math.ceil(0.8*(L9 + number))
    train_data = D[0:index, :, :, :]
    train_labels = labels[0:index]
    
    test_data = D[index:L9 + number, :, :, :]
    test_labels = labels[index:L9 + number]
    
    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y, pkeep: 0.3})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0})
        train_acc = train_acc + a
        train_loss = train_loss + c        
        batch = batch + 1
        
    train_loss = train_loss/batch
    train_acc = train_acc/batch
    
    testdata = {X: test_data, Y:test_labels, pkeep: 1.0}
    a,c = sess.run([accuracy,cross_entropy], feed_dict = testdata)
    test_acc = a
    test_loss = c
    
    print("Epoch",epoch + 1,"Train Loss",train_loss,"Train Acc",train_acc)
    print("Epoch",epoch + 1,"Test Loss",test_loss,"Test Acc",test_acc)
    print("\n")
    #saver.save(session, 'weights_model') 
