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

pkeep = 0.5

#-----------Functions start here-----------#
def read_from_folder(filename, max_value, min_value):
    a = sorted(os.listdir(filename)) # Sorted list of files
    
    m = len(a)
    
    images = np.zeros(shape = (m, ROWS, COLS, 3))
    
    for i in range(m):
        b = a[i]
        c = os.path.join(filename, b) # Full path to read image
        
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

def get_next_batch(index, dataset, classes, batch_size):
    
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.int32)
    
    return(X_batch,Y_batch) 


#------------Read original images-----------#
filename_1 = 'gossiping/'
a1 = sorted(os.listdir(filename_1))
l1 = len(a1)
# Read Folder_2

filename_2 = 'isolation/'
a2 = sorted(os.listdir(filename_2))
l2 = len(a2)
L2 = l1 + l2

# Read Folder_3
filename_3 = 'laughing/'
a3 = sorted(os.listdir(filename_3))
l3 = len(a3)
L3 = l1 + l2 + l3

# Read Folder_4
filename_4 = 'pullinghair/'
a4 = sorted(os.listdir(filename_4))
l4 = len(a4)
L4 = l1 + l2 + l3 + l4

# Read Folder_5
filename_5 = 'punching/'
a5 = sorted(os.listdir(filename_5))
l5 = len(a5)
L5 = l1 + l2 + l3 + l4 + l5

# Read Folder_6
filename_6 = 'quarrel/'
a6 = sorted(os.listdir(filename_6))
l6 = len(a6)
L6 = l1 + l2 + l3 + l4 + l5 + l6

# Read Folder_7
filename_7 = 'slapping/'
a7 = sorted(os.listdir(filename_7))
l7 = len(a7)
L7 = l1 + l2 + l3 + l4 + l5 + l6 + l7

# Read Folder_8
filename_8 = 'stabbing/'
a8 = sorted(os.listdir(filename_8))
l8 = len(a8)
L8 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8

# Read Folder_9
filename_9 = 'strangle/'
a9 = sorted(os.listdir(filename_9))
l9 = len(a9)
L9 = l1 + l2 + l3 + l4 + l5 + l6 + l7 + l8 + l9

# Read Standford 40 Images
filename_10 = 'stanford/'
a10 = sorted(os.listdir(filename_10))
l10 = len(a10)
number = L9
L10 = L9 + number
L = np.array([l1, L2, L3, L4, L5, L6, L7, L8, L9, L10], dtype = np.int32)

total_number = (4*L10)

# Initialize the dataset and label container
D = np.zeros(shape = (total_number, ROWS, COLS, 3)) 
labels = np.zeros(shape = (total_number)) 

# Read original images
D[0:l1, :, :, :] = read_from_folder(filename_1, 1, 0)
labels[0:l1] = 1

D[l1:L2, :, :, :] = read_from_folder(filename_2, 1, 0)
labels[l1:L2] = 2

D[L2:L3, :, :, :] = read_from_folder(filename_3, 1, 0)
labels[L2:L3] = 3

D[L3:L4, :, :, :] = read_from_folder(filename_4, 1, 0)
labels[L3:L4] = 4

D[L4:L5, :, :, :] = read_from_folder(filename_5, 1, 0)
labels[L4:L5] = 5

D[L5:L6, :, :, :] = read_from_folder(filename_6, 1, 0)
labels[L5:L6] = 6

D[L6:L7, :, :, :] = read_from_folder(filename_7, 1, 0)
labels[L6:L7] = 7

D[L7:L8, :, :, :] = read_from_folder(filename_8, 1, 0)
labels[L7:L8] = 8

D[L8:L9, :, :, :] = read_from_folder(filename_9, 1, 0)
labels[L8:L9] = 9

D[L9:L10, :, :, :] = read_nonbullying(filename_10, number, 1, 0)
labels[L9:L10] = 0

#---------------Data Augmentation------------------#
D[L10:(2*L10), :, :, :] = flip_images(D[0:L10, :, :, :]) # Flip images horizontally
D[(2*L10):(3*L10),:,:,:] = jitter(D[0:L10, :, :, :], 0.25) # Brightness jitter

D[(3*L10):(4*L10),:,:,:] = flip_images(D[0:L10, :, :, :]) # Flip + Additive Gaussian noise
D[(3*L10):(4*L10),:,:,:] = add_noise(D[(3*L10):(4*L10), :, :, :], 0, 0.25)

# Augment labels
augment_labels(labels, L10, L) # augment labels for flipped images

augment_labels(labels, (2*L10), L) # augment labels for jitter images

augment_labels(labels, (3*L10), L) # augment labels for flipped + noise images
# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
Y = tf.placeholder(tf.int32,[None])
depth = 10 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
# lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# layer 1 weights initialization
#filters_1_1 = filters_1_2 = 64
filters_1_1 = filters_1_2 = 16
W1_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, 3, filters_1_1], stddev = 0.1))
B1_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_1_1]))

W1_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_1_1, filters_1_2], stddev = 0.1))
B1_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_1_2]))

# layer 2 weights initialization
#filters_2_1 = filters_2_2 = 128
filters_2_1 = filters_2_2 = 32
W2_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_1_2, filters_2_1], stddev = 0.1))
B2_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_2_1]))

W2_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_2_1, filters_2_2], stddev = 0.1))
B2_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_2_2]))

# layer 3 weights initialization
#filters_3_1 = filters_3_2 = 256
filters_3_1 = filters_3_2 = 64
W3_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_2_2, filters_3_1], stddev = 0.1))
B3_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_3_1]))

W3_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_3_1, filters_3_2], stddev = 0.1))
B3_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_3_2]))

# layer 4 weights initialization
#filters_4_1 = filters_4_2 = 512
filters_4_1 = filters_4_2 = 128
W4_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_3_2, filters_4_1], stddev = 0.1))
B4_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_4_1]))

W4_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_4_1, filters_4_2], stddev = 0.1))
B4_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_4_2]))

# layer 5 weights initialization
#filters_5_1 = filters_5_2 = 512
filters_5_1 = filters_5_2 = 128
W5_1 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_4_2, filters_5_1], stddev = 0.1))
B5_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_5_1]))

W5_2 = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_5_1, filters_5_2], stddev = 0.1))
B5_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_5_2]))

#fc = 4096
fc = 200
# Fully Connected Layer Weight Initialization
WFC_1 = tf.Variable(tf.truncated_normal(shape = [7 * 7 * filters_5_2, fc], stddev = 0.1))
BFC_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc]))


# Fully Connected Layer Weight Initialization
WFC_2 = tf.Variable(tf.truncated_normal(shape = [fc, fc], stddev = 0.1))
BFC_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc]))

# Outut Layer Weight Initialization
WO = tf.Variable(tf.truncated_normal(shape = [fc, 10], stddev = 0.1))
BO = tf.Variable(tf.constant(0.1, tf.float32, shape = [10]))



# layer 1 network feed - forward part // X => 4D input tensor
Y1_1 = tf.nn.relu(tf.nn.conv2d(X, W1_1, strides = [1,1,1,1], padding = 'SAME') + B1_1)
Y1_2 = tf.nn.relu(tf.nn.conv2d(Y1_1, W1_2, strides = [1,1,1,1], padding = 'SAME') + B1_2)
Y1_out = tf.nn.max_pool(Y1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')



# layer 2 network feed - forward part // X => 4D input tensor
Y2_1 = tf.nn.relu(tf.nn.conv2d(Y1_out, W2_1, strides = [1,1,1,1], padding = 'SAME') + B2_1)
Y2_2 = tf.nn.relu(tf.nn.conv2d(Y2_1, W2_2, strides = [1,1,1,1], padding = 'SAME') + B2_2)
Y2_out = tf.nn.max_pool(Y2_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')



# layer 3 network feed - forward part // X => 4D input tensor
Y3_1 = tf.nn.relu(tf.nn.conv2d(Y2_out, W3_1, strides = [1,1,1,1], padding = 'SAME') + B3_1)
Y3_2 = tf.nn.relu(tf.nn.conv2d(Y3_1, W3_2, strides = [1,1,1,1], padding = 'SAME') + B3_2)
Y3_out = tf.nn.max_pool(Y3_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


# layer 4 network feed - forward part // X => 4D input tensor
Y4_1 = tf.nn.relu(tf.nn.conv2d(Y3_out, W4_1, strides = [1,1,1,1], padding = 'SAME') + B4_1)
Y4_2 = tf.nn.relu(tf.nn.conv2d(Y4_1, W4_2, strides = [1,1,1,1], padding = 'SAME') + B4_2)
Y4_out = tf.nn.max_pool(Y4_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')



# layer 5 network feed - forward part // X => 4D input tensor
Y5_1 = tf.nn.relu(tf.nn.conv2d(Y4_out, W5_1, strides = [1,1,1,1], padding = 'SAME') + B5_1)
Y5_2 = tf.nn.relu(tf.nn.conv2d(Y5_1, W5_2, strides = [1,1,1,1], padding = 'SAME') + B5_2)
Y5_out = tf.nn.max_pool(Y5_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

# FC Layers
YY = tf.reshape(Y5_out, shape = [-1, 7 * 7 * filters_5_2])
Y6 = tf.nn.relu(tf.matmul(YY, WFC_1) + BFC_1)
Y6_drop = tf.nn.dropout(Y6, keep_prob = pkeep)

Y7 = tf.nn.relu(tf.matmul(Y6_drop, WFC_2) + BFC_2)
Y7_drop = tf.nn.dropout(Y7, keep_prob = pkeep)


Y_logits= tf.matmul(Y7_drop, WO) + BO
Y_pred = tf.nn.softmax(Y_logits)

cross_entropy = tf.losses.sparse_softmax_cross_entropy(Y, Y_logits)
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


batch_size = 64 # 32 // Change code to check if code is in train/test

epochs = 125

total = 5000

train_loss = np.zeros(shape=(total))
train_acc = np.zeros(shape=(total))

test_loss = np.zeros(shape=(total))
test_acc = np.zeros(shape=(total))

iters1 = 0

for epoch in range(epochs):
            
    if epoch == 0:
        indices = np.arange(2*L10)
        np.random.shuffle(indices)
        index = math.floor(0.67*(2*L10))
        train_index = indices[0:index]
        test_index = indices[index:2*L10]
    else:
        np.random.shuffle(train_index)


    train_data = D[train_index, :, :, :]
    train_labels = labels[train_index]
    
    test_data = D[test_index, :, :, :]
    test_labels = labels[test_index]

    number_tst = (2*L10) - index # Number of testing images    

    # Testing images also need to fed in batches to avoid "Tensor out of memory error"
    # Hence loss and accuracy over test dataset is reported in steps

    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y, pkeep: 0.7})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0})
        train_acc[iters1] = a
        train_loss[iters1] = c   

        # Save model after each iteration
        #save_path = saver.save(sess,"E:/model_10categories")

        # Test calculation initialization
        acc = 0
        loss = 0
        
        # Line 569 - 578 => Testing loss # tf.reduce_mean is used hence there is no need to compute the average again
        for i in range(0, number_tst, batch_size):
            # start, stop, step
            batch_Xtst, batch_Ytst = get_next_batch(i, test_data, test_labels, batch_size)
            a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})
            acc = acc + a
            loss = loss + c
            
        
        test_acc[iters1] = (acc * batch_size)/ number_tst
        test_loss[iters1] = (loss * batch_size)/ number_tst
        
        # Print model train loss and accuracy after each iteration
        print("Iteration",iters1,"Train Loss",train_loss[iters1],"Train Acc",train_acc[iters1]) 
        print("Iteration",iters1,"Test Loss",test_loss[iters1],"Test Acc",test_acc[iters1]) 
        print("\n")
        iters1 = iters1 + 1 # update count of iterations 
        
np.savez('CPSC_8810/plots/iterations_plots.npz',name1 = train_acc, name2 = train_loss, name3 = test_acc, name4 = test_loss, name5 = iters1)
