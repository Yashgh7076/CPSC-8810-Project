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
    
def crop(dataset, topleft, bottomleft, topright, bottomright):
    
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
   


def get_next_batch(index, dataset, classes, batch_size):
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.int32)
    
    return(X_batch,Y_batch) 

# Code starts here
# Read Folder_1



filename_1 = 'gossiping/'
a1 = sorted(os.listdir(filename_1))
l1 = len(a1)
# Read Folder_2

filename_2 = 'isolation/'
a2 = sorted(os.listdir(filename_2))
l2 = len(a2)
L2 = 4*l1 + l2

# Read Folder_3
filename_3 = 'laughing/'
a3 = sorted(os.listdir(filename_3))
l3 = len(a3)
L3 = 4*l1 + 4*l2 + l3

# Read Folder_4
filename_4 = 'pullinghair/'
a4 = sorted(os.listdir(filename_4))
l4 = len(a4)
L4 = 4*l1 + 4*l2 + 4*l3 + l4

# Read Folder_5
filename_5 = 'punching/'
a5 = sorted(os.listdir(filename_5))
l5 = len(a5)
L5 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + l5

# Read Folder_6
filename_6 = 'quarrel/'
a6 = sorted(os.listdir(filename_6))
l6 = len(a6)
L6 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + l6

# Read Folder_7
filename_7 = 'slapping/'
a7 = sorted(os.listdir(filename_7))
l7 = len(a7)
L7 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + 4*l6 + l7

# Read Folder_8
filename_8 = 'stabbing/'
a8 = sorted(os.listdir(filename_8))
l8 = len(a8)
L8 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + 4*l6 + 4*l7 + l8

# Read Folder_9
filename_9 = 'strangle/'
a9 = sorted(os.listdir(filename_9))
l9 = len(a9)
L9 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + 4*l6 + 4*l7 + 4*l8 + l9

# Read Standford 40 Images
filename_10 = 'stanford/'
a10 = sorted(os.listdir(filename_10))
l10 = len(a10)
number = L9
L10 = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + 4*l6 + 4*l7 + 4*l8 + 4*l9 + number

# Initialize the dataset and label container
D = np.zeros(shape = (4*(L9 + number), ROWS, COLS, 3))
D_flip = np.zeros(shape = (L9 + number, ROWS, COLS, 3))
labels = np.zeros(shape = (4*(L9 + number)))

D[0:l1, :, :, :] = read_from_folder(filename_1, 1, 0)

D_flip = flip_images(D[0:l1, :, :, :])
D_noise = add_noise(D[0:l1, :, :, :], 0, 0.25)
D_crop = crop(D[0:l1, :, :, :], 25, 25, 25, 25)

D[l1:(2*l1),:,:,:] = D_flip
D[(2*l1):(3*l1),:,:,:] = D_noise
D[(3*l1):(4*l1),:,:,:] = D_crop

labels[0:l1] = 1
labels[l1:(2*l1)] = 1
labels[(2*l1):(3*l1)] = 1
labels[(3*l1):(4*l1)] = 1


D[(4*l1):L2, :, :, :] = read_from_folder(filename_2 , 1, 0)
D_flip = flip_images(D[(4*l1):L2, :, :, :])
D_noise = add_noise(D[(4*l1):L2, :, :, :], 0, 0.25)
D_crop = crop(D[(4*l1):L2, :, :, :], 25, 25, 25, 25)

D[L2:L2 + l2,:,:,:] = D_flip # L2 = 4l1 + l2
D[L2 + l2:L2 + (2*l2),:,:,:] = D_noise
D[L2 + (2*l2): L2 + (3*l2),:,:,:] = D_crop

labels[4*l1:L2] = 2
labels[L2:L2 + l2] = 2
labels[L2 + l2:L2 + (2*l2)] = 2
labels[L2 + (2*l2): L2 + (3*l2)] = 2


D[(4*l1) + (4*l2):L3, :, :, :] = read_from_folder(filename_3, 1 ,0)
D_flip = flip_images(D[(4*l1) + (4*l2):L3, :, :, :])
D_noise = add_noise(D[(4*l1) + (4*l2):L3, :, :, :], 0, 0.25)
D_crop = crop(D[(4*l1) + (4*l2):L3, :, :, :], 25, 25, 25, 25)

D[L3:L3 + l3,:,:,:] = D_flip # L3 = 4l1 + 4l2 + l3
D[L3 + l3:L3 + (2*l3),:,:,:] = D_noise
D[L3 + (2*l3) : L3 + (3*l3),:,:,:] = D_crop

labels[(4*l1) + (4*l2):L3] = 3
labels[L3:L3 + l3] = 3
labels[L3 + l3:L3 + (2*l3)] = 3
labels[L3 + (2*l3): L3 + (3*l3)] = 3



D[(4*l1) + (4*l2) + (4*l3):L4, :, :, :] = read_from_folder(filename_4, 1, 0)  # L4 = 4l1 + 4l2 + 4l3 + l4
D_flip = flip_images(D[(4*l1) + (4*l2) + (4*l3):L4, :, :, :])
D_noise = add_noise(D[(4*l1) + (4*l2) + (4*l3):L4, :, :, :], 0, 0.25)
D_crop = crop(D[(4*l1) + (4*l2) + (4*l3):L4, :, :, :], 25, 25, 25, 25)

D[L4:L4+l4,:,:,:] = D_flip # L4 = 4l1 + 4l2 + 4l3 + l4
D[L4 + l4:L4 + (2*l4),:,:,:] = D_noise
D[L4 + (2*l4) : L4 + (3*l4),:,:,:] = D_crop


labels[(4*l1) + (4*l2) + (4*l3):L4] = 4
labels[L4:L4 + l4] = 4
labels[L4 + l4:L4 + (2*l4)] = 4
labels[L4 + (2*l4): L4 + (3*l4)] = 4


temp_index = (4*l1) + (4*l2) + (4*l3) + (4*l4) # Create a temporary index of the number of images 
D[temp_index:L5, :, :, :] = read_from_folder(filename_5, 1 , 0)  # L5 = 4l1 + 4l2 + 4l3 + 4l4 + l5 = temp_index + l5

D_flip = flip_images(D[temp_index:L5, :, :, :])
D_noise = add_noise(D[temp_index:L5, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L5, :, :, :], 25, 25, 25, 25)

D[L5:L5+l5,:,:,:] = D_flip
D[L5 + l5:L5 + (2*l5),:,:,:] = D_noise
D[L5 + (2*l5) : L5 + (3*l5),:,:,:] = D_crop

labels[temp_index:L5] = 5
labels[L5:L5 + l5] = 5
labels[L5 + l5:L5 + (2*l5)] = 5
labels[L5 + (2*l5): L5 + (3*l5)] = 5

temp_index = 4*(l1 + l2 + l3 + l4 + l5)
D[temp_index:L6, :, :, :] = read_from_folder(filename_6, 1, 0)

D_flip = flip_images(D[temp_index:L6, :, :, :])
D_noise = add_noise(D[temp_index:L6, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L6, :, :, :], 25, 25, 25, 25)

D[L6:L6+l6,:,:,:] = D_flip
D[L6 + l6:L6 + (2*l6),:,:,:] = D_noise
D[L6+(2*l6):L6 + (3*l6),:,:,:] = D_crop

labels[temp_index:L6] = 6
labels[L6:L6 + l6] = 6
labels[L6 + l6:L6 + (2*l6)] = 6
labels[L6 + (2*l6): L6 + (3*l6)] = 6

temp_index = 4*(l1 + l2 + l3 + l4 + l5 + l6)
D[temp_index:L7, :, :, :] = read_from_folder(filename_7, 1, 0)

D_flip = flip_images(D[temp_index:L7, :, :, :])
D_noise = add_noise(D[temp_index:L7, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L7, :, :, :], 25, 25, 25, 25)

D[L7:L7+l7,:,:,:] = D_flip
D[L7 + l7:L7 + (2*l7),:,:,:] = D_noise
D[L7 + (2*l7):L7 + (3*l7),:,:,:] = D_crop

labels[temp_index:L7] = 7
labels[L7:L7 + l7] = 7
labels[L7 + l7:L7 + (2*l7)] = 7
labels[L7 + (2*l7): L7 + (3*l7)] = 7

temp_index = 4*(l1 + l2 + l3 + l4 + l5 + l6 + l7)
D[temp_index:L8, :, :, :] = read_from_folder(filename_8, 1, 0)
D_flip = flip_images(D[temp_index:L8, :, :, :])
D_noise = add_noise(D[temp_index:L8, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L8, :, :, :], 25, 25, 25, 25)

D[L8:L8 + l8,:,:,:] = D_flip
D[L8 + l8:L8 + (2*l8),:,:,:] = D_noise
D[L8 + (2*l8):L8 + (3*l8),:,:,:] = D_crop

labels[temp_index:L8] = 8
labels[L8:L8 + l8] = 8
labels[L8 + l8:L8 + (2*l8)] = 8
labels[L8 + (2*l8): L8 + (3*l8)] = 8

temp_index = 4*(l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8)
D[temp_index:L9, :, :, :] = read_from_folder(filename_9, 1, 0)
D_flip = flip_images(D[temp_index:L9, :, :, :])
D_noise = add_noise(D[temp_index:L9, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L9, :, :, :], 25, 25, 25, 25)

D[L9:L9 + l9,:,:,:] = D_flip
D[L9 + l9:L9 + (2*l9),:,:,:] = D_noise
D[L9 + (2*l9):L9 + (3*l9),:,:,:] = D_crop

labels[temp_index:L9] = 9
labels[L9:L9 + l9] = 9
labels[L9 + l9:L9 + (2*l9)] = 9
labels[L9 + (2*l9): L9 + (3*l9)] = 9

temp_index = 4*l1 + 4*l2 + 4*l3 + 4*l4 + 4*l5 + 4*l6 + 4*l7 + 4*l8 + 4*l9
D[temp_index:L10, :, :, :] = read_nonbullying(filename_10, number, 1, 0)


D_flip = flip_images(D[temp_index:L10, :, :, :])
D_noise = add_noise(D[temp_index:L10, :, :, :], 0, 0.25)
D_crop = crop(D[temp_index:L10, :, :, :], 25, 25, 25, 25)

D[L10:L10 + number,:,:,:] = D_flip
D[L10 + number :L10 + (2*number),:,:,:] = D_noise
D[L10 + (2*number) : L10+ (3*number),:,:,:] = D_crop

labels[temp_index:L10] = 0
labels[L10:L10 + number] = 0
labels[L10 + number:L10 + (2*number)] = 0
labels[L10 + (2*number): L10 + (3*number)] = 0


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
saver = tf.train.Saver()

batch_size = 64 # 32 // Change code to check if code is in train/test

epochs = 125

total = 50000

train_loss = np.zeros(shape=(total))
train_acc = np.zeros(shape=(total))

test_loss = np.zeros(shape=(total))
test_acc = np.zeros(shape=(total))

iters = 0
iters1 = 0
iters2 = 0

for epoch in range(epochs):
            
    if epoch == 0:
        indices = np.arange(4*(L9 + number))
        np.random.shuffle(indices)
        index = math.floor(0.67*(4*(L9 + number)))
        train_index = indices[0:index]
        test_index = indices[index:(4*(L9 + number))]
    else:
        np.random.shuffle(train_index)


    train_data = D[train_index, :, :, :]
    train_labels = labels[train_index]
    
    test_data = D[test_index, :, :, :]
    test_labels = labels[test_index]


    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y, pkeep: 0.7})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0})
        train_acc[iters1] = a
        train_loss[iters1] = c        
        iters1 = iters1 + 1

        # Save model after each iteration
        save_path = saver.save(sess,"D:/model_10categories")
        
        
    #train_loss[epoch] = train_loss[epoch]/batch
    #train_acc[epoch] = train_acc[epoch]/batch
    
    
    #testdata = {X: test_data, Y:test_labels, pkeep: 1.0}
    #a,c = sess.run([accuracy,cross_entropy], feed_dict = testdata)
    #test_acc = a
    #test_loss = c

    # batch_tst = 1
    number_tst = (4*(L9 + number)) - index

    for i in range(0, number_tst, batch_size):
        #start, stop, step
        batch_Xtst, batch_Ytst = get_next_batch(i, test_data, test_labels, batch_size)
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})
        test_acc[iters2] = a
        test_loss[iters2] = c        
        iters2 = iters2 + 1
    

    #test_loss[epoch] = test_loss[epoch]/batch_tst
    #test_acc[epoch] = test_acc[epoch]/batch_tst

   

    # Print model train loss and accuracy after each iteration
    print("Iteration",iters,"Train Loss",train_loss[iters],"Train Acc",train_acc[iters]) 
    print("Iteration",iters,"Test Loss",test_loss[iters],"Test Acc",test_acc[iters]) 
    print("\n")
   
    
np.savez('D:/plots/iterations_plots.npz',name1 = train_acc, name2 = train_loss, name3 = test_acc, name4 = test_loss)

#np.savez('CPSC_8810/plots/10categoriesnew_plots.npz',name1 = train_acc, name2 = train_loss, name3 = test_acc, name4 = test_loss)