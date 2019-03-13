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
tf.set_random_seed(0)

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
number = 1000

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
# lr = .placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# Specifytf parameters of the NN model & training 
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
saver = tf.train.Saver()

#indices = np.arange(L9 + number)
#np.random.shuffle(indices)
#D = D[indices, :, :, :]
#labels = labels[indices]
#train_data = D[0:indices, :, :, :]
#train_labels = labels[0:indices]
#number =L9 + number

batch_size = 64
batch_tst = 1
#number =L9
#number_tst = (L9 + number) - index
#category = np.zeros(shape = (number_tst))
count_g = np.zeros(shape = (10))
count_i = np.zeros(shape = (10))
count_l = np.zeros(shape = (10))
count_ph = np.zeros(shape = (10))
count_pun = np.zeros(shape = (10))
count_q = np.zeros(shape = (10))
count_sl = np.zeros(shape = (10))
count_stab = np.zeros(shape = (10))
count_stran = np.zeros(shape = (10))

count_b = 0
count_nb = np.zeros(shape = (10))


with tf.Session() as sess:
    saver.restore(sess,"CPSC_8810/model_10cat/model_10categories")
    #saver.restore(sess,"model_9cat/model_9categories")
    for i in range(0, L9 + number, batch_size):
        #start, stop, step
        batch_Xtst, batch_Ytst = get_next_batch(i, D , labels, batch_size)
        #a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})
        prediction = sess.run([Y_pred], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})      
        batch_tst = batch_tst + 1
        pred = np.array(prediction)
        pred = np.reshape(pred,(pred.shape[1],pred.shape[2]))
        #print(pred)
        #category = np.argmax(pred)
        #print(category)
        #print(np.argmax(pred, axis = 1))
        for j in range(pred.shape[0]):
            category = np.argmax(pred[j], axis = 0)
            #print(category)
            if(category == 0):
                if(batch_Ytst[j] == 0):
                    count_nb[9] = count_nb[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_nb[0] = count_nb[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_nb[1] = count_nb[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_nb[2] = count_nb[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_nb[3] = count_nb[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_nb[4] = count_nb[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_nb[5] = count_nb[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_nb[6] = count_nb[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_nb[7] = count_nb[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_nb[8] = count_nb[8] + 1
            if(category == 1):
                if(batch_Ytst[j] == 0):
                    count_g[9] = count_g[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_g[0] = count_g[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_g[1] = count_g[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_g[2] = count_g[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_g[3] = count_g[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_g[4] = count_g[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_g[5] = count_g[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_g[6] = count_g[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_g[7] = count_g[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_g[8] = count_g[8] + 1
            if(category == 2):
                if(batch_Ytst[j] == 0):
                    count_i[9] = count_i[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_i[0] = count_i[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_i[1] = count_i[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_i[2] = count_i[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_i[3] = count_i[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_i[4] = count_i[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_i[5] = count_i[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_i[6] = count_i[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_i[7] = count_i[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_i[8] = count_i[8] + 1
            if(category == 3):
                if(batch_Ytst[j] == 0):
                    count_l[9] = count_l[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_l[0] = count_l[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_l[1] = count_l[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_l[2] = count_l[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_l[3] = count_l[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_l[4] = count_l[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_l[5] = count_l[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_l[6] = count_l[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_l[7] = count_l[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_l[8] = count_l[8] + 1
            if(category == 4):
                if(batch_Ytst[j] == 0):
                    count_ph[9] = count_ph[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_ph[0] = count_ph[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_ph[1] = count_ph[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_ph[2] = count_ph[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_ph[3] = count_ph[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_ph[4] = count_ph[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_ph[5] = count_ph[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_ph[6] = count_ph[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_ph[7] = count_ph[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_ph[8] = count_ph[8] + 1
            if(category == 5):
                if(batch_Ytst[j] == 0):
                    count_pun[9] = count_pun[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_pun[0] = count_pun[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_pun[1] = count_pun[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_pun[2] = count_pun[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_pun[3] = count_pun[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_pun[4] = count_pun[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_pun[5] = count_pun[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_pun[6] = count_pun[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_pun[7] = count_pun[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_pun[8] = count_pun[8] + 1
            if(category == 6):
                if(batch_Ytst[j] == 0):
                    count_q[9] = count_q[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_q[0] = count_q[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_q[1] = count_q[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_q[2] = count_q[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_q[3] = count_q[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_q[4] = count_q[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_q[5] = count_q[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_q[6] = count_q[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_q[7] = count_q[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_q[8] = count_q[8] + 1
            if(category == 7):
                if(batch_Ytst[j] == 0):
                    count_sl[9] = count_sl[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_sl[0] = count_sl[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_sl[1] = count_sl[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_sl[2] = count_sl[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_sl[3] = count_sl[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_sl[4] = count_sl[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_sl[5] = count_sl[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_sl[6] = count_sl[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_sl[7] = count_sl[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_sl[8] = count_sl[8] + 1
            if(category == 8):
                if(batch_Ytst[j] == 0):
                    count_stab[9] = count_stab[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_stab[0] = count_stab[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_stab[1] = count_stab[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_stab[2] = count_stab[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_stab[3] = count_stab[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_stab[4] = count_stab[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_stab[5] = count_stab[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_stab[6] = count_stab[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_stab[7] = count_stab[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_stab[8] = count_stab[8] + 1
            if(category == 9):
                if(batch_Ytst[j] == 0):
                    count_stran[9] = count_stran[9] + 1
                elif(batch_Ytst[j] == 1):
                    count_stran[0] = count_stran[0] + 1
                elif(batch_Ytst[j] == 2):
                    count_stran[1] = count_stran[1] + 1
                elif(batch_Ytst[j] == 3):
                    count_stran[2] = count_stran[2] + 1
                elif(batch_Ytst[j] == 4):
                    count_stran[3] = count_stran[3] + 1
                elif(batch_Ytst[j] == 5):
                    count_stran[4] = count_stran[4] + 1
                elif(batch_Ytst[j] == 6):
                    count_stran[5] = count_stran[5] + 1
                elif(batch_Ytst[j] == 7):
                    count_stran[6] = count_stran[6] + 1
                elif(batch_Ytst[j] == 8):
                    count_stran[7] = count_stran[7] + 1
                elif(batch_Ytst[j] == 9):
                    count_stran[8] = count_stran[8] + 1
    
    print("Gossiping",count_g[0], count_g[1], count_g[2], count_g[3], count_g[4], count_g[5], count_g[6], count_g[7], count_g[8], count_g[9])
    print("Isolation",count_i[0], count_i[1], count_i[2], count_i[3], count_i[4], count_i[5], count_i[6], count_i[7], count_i[8], count_i[9])
    print("Laughing",count_l[0], count_l[1], count_l[2], count_l[3], count_l[4], count_l[5], count_l[6], count_l[7], count_l[8], count_l[9])
    print("Pulling Hair",count_ph[0], count_ph[1], count_ph[2], count_ph[3], count_ph[4], count_ph[5], count_ph[6], count_ph[7], count_ph[8], count_ph[9])
    print("Punching",count_pun[0], count_pun[1], count_pun[2], count_pun[3], count_pun[4], count_pun[5], count_pun[6], count_pun[7], count_pun[8], count_pun[9])
    print("Quarrel",count_q[0], count_q[1], count_q[2], count_q[3], count_q[4], count_q[5], count_q[6], count_q[7], count_q[8], count_q[9])
    print("Slapping",count_sl[0], count_sl[1], count_sl[2], count_sl[3], count_sl[4], count_sl[5], count_sl[6], count_sl[7], count_sl[8], count_sl[9])
    print("Stabbing",count_stab[0], count_stab[1], count_stab[2], count_stab[3], count_stab[4], count_stab[5], count_stab[6], count_stab[7], count_stab[8], count_stab[9])
    print("Strangle",count_stran[0], count_stran[1], count_stran[2], count_stran[3], count_stran[4], count_stran[5], count_stran[6], count_stran[7], count_stran[8], count_stran[9])
    print("Non_bullying", count_nb[0], count_nb[1], count_nb[2], count_nb[3], count_nb[4], count_nb[5], count_nb[6], count_nb[7], count_nb[8], count_nb[9])