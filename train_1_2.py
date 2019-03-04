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
pkeep = 0.5
window = 9

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
number = 200

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
labels[L9:L9 + number] = 10

# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
Y = tf.placeholder(tf.int32,[None])
depth = 10 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
# lr = tf.placeholder(tf.float32)
# pkeep = tf.placeholder(tf.float32)

# Initialize Weights

# layer 1 weights initialization
filters_1_1 = 32 # filters_1_2
W1_1 = tf.Variable(tf.truncated_normal(shape = [window, window, 3, filters_1_1], stddev = 0.1))
B1_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_1_1]))

# layer 2 weights initialization
filters_2_1 = 16 # filters_2_2 = 8
W2_1 = tf.Variable(tf.truncated_normal(shape = [window, window, filters_1_1, filters_2_1], stddev = 0.1))
B2_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_2_1]))

# layer 3 weights initialization
filters_3_1 = 8 # filters_3_2 = 16
W3_1 = tf.Variable(tf.truncated_normal(shape = [window, window, filters_2_1, filters_3_1], stddev = 0.1))
B3_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_3_1]))

# layer 4 weights initialization
filters_4_1 = 4 # filters_4_2 = 32
W4_1 = tf.Variable(tf.truncated_normal(shape = [window, window, filters_3_1, filters_4_1], stddev = 0.1))
B4_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_4_1]))

# layer 5 weights initialization
filters_5_1 = 4 # filters_5_2 = 32
W5_1 = tf.Variable(tf.truncated_normal(shape = [window, window, filters_4_1, filters_5_1], stddev = 0.1))
B5_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_5_1]))

fc = 8 #4096
# Fully Connected Layer Weight Initialization
WFC_1 = tf.Variable(tf.truncated_normal(shape = [7 * 7 * filters_5_1, fc], stddev = 0.1))
BFC_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc]))

# Fully Connected Layer Weight Initialization
WFC_2 = tf.Variable(tf.truncated_normal(shape = [fc, fc], stddev = 0.1))
BFC_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc]))

# Outut Layer Weight Initialization
WO = tf.Variable(tf.truncated_normal(shape = [fc, 10], stddev = 0.1))
BO = tf.Variable(tf.constant(0.1, tf.float32, shape = [10]))

# Create model Sequence is CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
# layer 1 network feed - forward part // X => 4D input tensor
Y1_1 = tf.nn.relu(tf.nn.conv2d(X, W1_1, strides = [1,1,1,1], padding = 'SAME') + B1_1)
Y1_1_drop = tf.nn.max_pool(Y1_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

mean1_1, var1_1 = tf.nn.moments(Y1_1_drop, axes = [0, 1, 2])
Y1_1_bn = tf.nn.batch_normalization(Y1_1_drop, mean1_1, var1_1, scale = 1, offset = 0.01, variance_epsilon = 0.001)

# layer 2 network feed - forward part // X => 4D input tensor
Y2_1 = tf.nn.relu(tf.nn.conv2d(Y1_1_bn, W2_1, strides = [1,1,1,1], padding = 'SAME') + B2_1)
Y2_1_drop = tf.nn.max_pool(Y2_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

mean2_1, var2_1 = tf.nn.moments(Y2_1_drop, axes = [0, 1, 2])
Y2_1_bn = tf.nn.batch_normalization(Y2_1_drop, mean2_1, var2_1, scale = 1, offset = 0.01, variance_epsilon = 0.001)

# layer 3 network feed - forward part // X => 4D input tensor
Y3_1 = tf.nn.relu(tf.nn.conv2d(Y2_1_bn, W3_1, strides = [1,1,1,1], padding = 'SAME') + B3_1)
Y3_1_drop = tf.nn.max_pool(Y3_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

mean3_1, var3_1 = tf.nn.moments(Y3_1_drop, axes = [0, 1, 2])
Y3_1_bn = tf.nn.batch_normalization(Y3_1_drop, mean3_1, var3_1, scale = 1, offset = 0.01, variance_epsilon = 0.001)

# layer 4 network feed - forward part // X => 4D input tensor
Y4_1 = tf.nn.relu(tf.nn.conv2d(Y3_1_bn, W4_1, strides = [1,1,1,1], padding = 'SAME') + B4_1)
Y4_1_drop = tf.nn.max_pool(Y4_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

mean4_1, var4_1 = tf.nn.moments(Y4_1_drop, axes = [0, 1, 2])
Y4_1_bn = tf.nn.batch_normalization(Y4_1_drop, mean4_1, var4_1, scale = 1, offset = 0.01, variance_epsilon = 0.001)

# layer 5 network feed - forward part // X => 4D input tensor
Y5_1 = tf.nn.relu(tf.nn.conv2d(Y4_1_bn, W5_1, strides = [1,1,1,1], padding = 'SAME') + B5_1)
Y5_1_drop = tf.nn.max_pool(Y5_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

mean5_1, var5_1 = tf.nn.moments(Y5_1_drop, axes = [0, 1, 2])
Y5_1_bn = tf.nn.batch_normalization(Y5_1_drop, mean5_1, var5_1, scale = 1, offset = 0.01, variance_epsilon = 0.001)

# FC Layers
YY = tf.reshape(Y5_1_bn, shape = [-1, 7 * 7 * filters_5_1])
Y6 = tf.nn.relu(tf.matmul(YY, WFC_1) + BFC_1)
Y6_drop = tf.nn.dropout(Y6, keep_prob = pkeep)

mean6, var6 = tf.nn.moments(Y6_drop, axes = [0])
Y6_bn = tf.nn.batch_normalization(Y6_drop, mean6, var6, scale = 1, offset = 0.01, variance_epsilon = 0.001)

Y7 = tf.nn.relu(tf.matmul(Y6_bn, WFC_2) + BFC_2)
Y7_drop = tf.nn.dropout(Y7, keep_prob = pkeep)

mean7, var7 = tf.nn.moments(Y7_drop, axes = [0])
Y7_bn = tf.nn.batch_normalization(Y7_drop, mean7, var7, scale = 1, offset = 0.01, variance_epsilon = 0.001)

Y_logits= tf.matmul(Y7_bn, WO) + BO
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

batch_size = 64

for epoch in range(100):
    batch = 1
    train_loss = 0
    train_acc = 0

    test_loss = 0
    test_acc = 0
    
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
        sess.run(train_step,feed_dict = {X: batch_X, Y: batch_Y})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y})
        train_acc = train_acc + a
        train_loss = train_loss + c
    
        testdata = {X: test_data, Y:test_labels}
        a,c = sess.run([accuracy,cross_entropy], feed_dict = testdata)
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
    #saver.save(session, 'weights_model') 
