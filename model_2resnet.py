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
window = 9
tf.set_random_seed(0)

#-----------Functions start here-----------#
def read_from_folder(filename):
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

#------------Read original images-----------#
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
number = L9
L10 = L9 + number
L = np.array([l1, L2, L3, L4, L5, L6, L7, L8, L9, L10], dtype = np.int32)

total_number = (4*L10)

# Initialize the dataset and label container
D = np.zeros(shape = (total_number, ROWS, COLS, 3)) 
labels = np.zeros(shape = (total_number)) 

# Read original images
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

D[L9:L10, :, :, :] = read_nonbullying(filename_10, number)
labels[L9:L10] = 0

#---------------Data Augmentation------------------#
D[L10:(2*L10), :, :, :] = flip_images(D[0:L10, :, :, :]) # Flip images horizontally
D[(2*L10):(3*L10),:,:,:] = jitter(D[0:L10, :, :, :], 0.25) # Brightness jitter

D[(3*L10):(4*L10),:,:,:] = flip_images(D[0:L10, :, :, :]) # Flip + Additive Gaussian noise
D[(3*L10):(4*L10),:,:,:] = add_noise(D[(3*L10):(4*L10), :, :, :], 0, 0.25)

D = standard_scaler(D)

# Augment labels
augment_labels(labels, L10, L) # augment labels for flipped images

augment_labels(labels, (2*L10), L) # augment labels for jitter images

augment_labels(labels, (3*L10), L) # augment labels for flipped + noise images

#--------------Model Starts Here--------------------#
# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
Y = tf.placeholder(tf.int32,[None])
depth = 10 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)
pkeep_conv = tf.placeholder(tf.float32)
#lambda_reg = tf.placeholder(tf.float32)
training = tf.placeholder_with_default(False, shape = ())

# Initialize weights
# 1 -> convolution layer // Image Size 224 x 224 x 3
filters_1conv= 16
W1_conv= tf.Variable(tf.truncated_normal(shape = [window, window, 3, filters_1conv], stddev = 0.1))
B1_conv = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_1conv]))

# 2 -> residual block // Image Size 112 x 112 X filters1_conv
filters_1res = filters_2res = 32
W1_res = tf.Variable(tf.truncated_normal(shape = [7, 7, filters_1conv, filters_1res], stddev = 0.1))
B1_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_1res]))

W2_res = tf.Variable(tf.truncated_normal(shape = [7, 7, filters_1res, filters_2res], stddev = 0.1))
B2_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_2res]))

# 3 -> residual block// Image Size 56 x 56 x filters_2res
filters_3res = filters_4res = 64
W3_res = tf.Variable(tf.truncated_normal(shape = [5, 5, filters_2res, filters_3res], stddev = 0.1))
B3_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_3res]))

W4_res = tf.Variable(tf.truncated_normal(shape = [5, 5, filters_3res, filters_4res], stddev = 0.1))
B4_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_4res]))

# 4 -> residual block// Image Size 28 x 28 x filters_4res
filters_5res = filters_6res = 128
W5_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_4res, filters_5res], stddev = 0.1))
B5_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_5res]))

W6_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_5res, filters_6res], stddev = 0.1))
B6_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_6res]))

# 5 -> residual block// Image Size 14 x 14 x filters_6res
filters_7res = filters_8res = 256
W7_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_6res, filters_7res], stddev = 0.1))
B7_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_7res]))

W8_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_7res, filters_8res], stddev = 0.1))
B8_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_8res]))

# 6 -> residual block// Image Size 7 x 7 x filters_8res
filters_9res = filters_10res = 512
W9_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_8res, filters_9res], stddev = 0.1))
B9_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_9res]))

W10_res = tf.Variable(tf.truncated_normal(shape = [3, 3, filters_9res, filters_10res], stddev = 0.1))
B10_res = tf.Variable(tf.constant(0.1, tf.float32, shape = [filters_10res]))

# 7 -> fully connected layers
fc_1 = 1000
fc_2 = 500
# Fully Connected Layer Weight Initialization
WFC_1 = tf.Variable(tf.truncated_normal(shape = [7 * 7 * filters_10res, fc_1], stddev = 0.1))
BFC_1 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc_1]))

WFC_2 = tf.Variable(tf.truncated_normal(shape = [fc_1, fc_2], stddev = 0.1))
BFC_2 = tf.Variable(tf.constant(0.1, tf.float32, shape = [fc_2]))

# Outut Layer Weight Initialization
WO = tf.Variable(tf.truncated_normal(shape = [fc_2, 10], stddev = 0.1))
BO = tf.Variable(tf.constant(0.1, tf.float32, shape = [10]))

#------------Model starts here------------#
# Input image 224 x 224 x 3
Y1_conv = tf.nn.conv2d(X, W1_conv, strides = [1,1,1,1], padding = 'SAME') + B1_conv
Y1_conv = tf.layers.batch_normalization(Y1_conv, training = training, momentum = 0.5)
Y1_conv = tf.nn.elu(Y1_conv)
Y1_conv = tf.nn.max_pool(Y1_conv, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y1_conv = tf.nn.dropout(Y1_conv, keep_prob = pkeep_conv)

# Input image 112 x 112 x filters_1conv
Y1_res = tf.nn.conv2d(Y1_conv, W1_res, strides =[1,1,1,1], padding = 'SAME') + B1_res
Y1_res = tf.layers.batch_normalization(Y1_res, training = training, momentum = 0.5)
Y1_res = tf.nn.elu(Y1_res) # No maxpooling within a resnet block

Y2_res = tf.nn.conv2d(Y1_res, W2_res, strides =[1,1,1,1], padding = 'SAME') + B2_res
Y2_res = Y2_res + tf.concat([Y1_conv, Y1_conv], 3) # Add input of resnet block
Y2_res = tf.layers.batch_normalization(Y2_res, training = training, momentum = 0.5)
Y2_res = tf.nn.elu(Y2_res)
Y2_res = tf.nn.max_pool(Y2_res, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y2_res = tf.nn.dropout(Y2_res, keep_prob = pkeep_conv)

# Input image 56 x 56 x filters_2res
Y3_res = tf.nn.conv2d(Y2_res, W3_res, strides =[1,1,1,1], padding = 'SAME') + B3_res
Y3_res = tf.layers.batch_normalization(Y3_res, training = training, momentum = 0.5)
Y3_res = tf.nn.elu(Y3_res) # No maxpooling within a resnet block

Y4_res = tf.nn.conv2d(Y3_res, W4_res, strides =[1,1,1,1], padding = 'SAME') + B4_res
Y4_res = Y4_res + tf.concat([Y2_res, Y2_res], 3) # Add input of resnet block
Y4_res = tf.layers.batch_normalization(Y4_res, training = training, momentum = 0.5)
Y4_res = tf.nn.elu(Y4_res)
Y4_res = tf.nn.max_pool(Y4_res, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y4_res = tf.nn.dropout(Y4_res, keep_prob = pkeep_conv)

# Input image 28 x 28 x filters_4res
Y5_res = tf.nn.conv2d(Y4_res, W5_res, strides =[1,1,1,1], padding = 'SAME') + B5_res
Y5_res = tf.layers.batch_normalization(Y5_res, training = training, momentum = 0.5)
Y5_res = tf.nn.elu(Y5_res) # No maxpooling within a resnet block

Y6_res = tf.nn.conv2d(Y5_res, W6_res, strides =[1,1,1,1], padding = 'SAME') + B6_res
Y6_res = Y6_res + tf.concat([Y4_res, Y4_res], 3) # Add input of resnet block
Y6_res = tf.layers.batch_normalization(Y6_res, training = training, momentum = 0.5)
Y6_res = tf.nn.elu(Y6_res)
Y6_res = tf.nn.max_pool(Y6_res, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y6_res = tf.nn.dropout(Y6_res, keep_prob = pkeep_conv)

# Input image 14 x 14 x filters_6res
Y7_res = tf.nn.conv2d(Y6_res, W7_res, strides =[1,1,1,1], padding = 'SAME') + B7_res
Y7_res = tf.layers.batch_normalization(Y7_res, training = training, momentum = 0.5)
Y7_res = tf.nn.elu(Y7_res) # No maxpooling within a resnet block

Y8_res = tf.nn.conv2d(Y7_res, W8_res, strides =[1,1,1,1], padding = 'SAME') + B8_res
Y8_res = Y8_res + tf.concat([Y6_res, Y6_res], 3) # Add input of resnet block
Y8_res = tf.layers.batch_normalization(Y8_res, training = training, momentum = 0.5)
Y8_res = tf.nn.elu(Y8_res)
Y8_res = tf.nn.max_pool(Y8_res, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y8_res = tf.nn.dropout(Y8_res, keep_prob = pkeep_conv)

# Input image 7 x 7 x filters_8res
Y9_res = tf.nn.conv2d(Y8_res, W9_res, strides =[1,1,1,1], padding = 'SAME') + B9_res
Y9_res = tf.layers.batch_normalization(Y9_res, training = training, momentum = 0.5)
Y9_res = tf.nn.elu(Y9_res) # No maxpooling within a resnet block

Y10_res = tf.nn.conv2d(Y9_res, W10_res, strides =[1,1,1,1], padding = 'SAME') + B10_res
Y10_res = Y10_res + tf.concat([Y8_res, Y8_res], 3) # Add input of resnet block
Y10_res = tf.layers.batch_normalization(Y10_res, training = training, momentum = 0.5)
Y10_res = tf.nn.elu(Y10_res)
#Y10_res = tf.nn.max_pool(Y10_res, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
Y10_res = tf.nn.dropout(Y10_res, keep_prob = pkeep_conv)

# Input image 7 x 7 x filters_10res
Y7 = tf.reshape(Y10_res, shape = [-1, 7 * 7 * filters_10res])
Y7 = tf.matmul(Y7, WFC_1) + BFC_1
Y7 = tf.nn.elu(Y7)
Y7= tf.nn.dropout(Y7, keep_prob = pkeep)

Y8 = tf.matmul(Y7, WFC_2) + BFC_2
Y8 = tf.nn.elu(Y8)
Y8 = tf.nn.dropout(Y8, keep_prob = pkeep)

Y_logits= tf.matmul(Y8, WO) + BO
Y_pred = tf.nn.softmax(Y_logits)

# Calculate loss
cross_entropy = tf.losses.softmax_cross_entropy(Y_onehot, Y_logits)
cross_entropy = tf.reduce_mean(cross_entropy)

# Calculate regularization
#reg = tf.nn.l2_loss(W1_conv) + tf.nn.l2_loss(W1_res) + tf.nn.l2_loss(W2_res) + tf.nn.l2_loss(W3_res) + tf.nn.l2_loss(W4_res) + tf.nn.l2_loss(W5_res) + tf.nn.l2_loss(W6_res)

# Calculate total loss
#loss = cross_entropy + lambda_reg*reg

# Calculate accuracy for each mini batch
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

epochs = 100

initial_learn_rate = 0.0001

#total = 700

#train_loss = np.zeros(shape=(total))
#train_acc = np.zeros(shape=(total))

#test_loss = np.zeros(shape=(total))
#test_acc = np.zeros(shape=(total))


f = open('model_2resnet_evaluation.txt','w')
f.write("Iteration Training_Loss Training_Acc Test_Loss Test_Acc \n")

iters1 = 0

for epoch in range(epochs):
            
    if epoch == 0:
        indices = np.arange(total_number)
        np.random.shuffle(indices)
        index = math.floor(0.67*(total_number))
        train_index = indices[0:index]
        test_index = indices[index:total_number]
    else:
        np.random.shuffle(train_index)

    train_data = D[train_index, :, :, :]
    train_labels = labels[train_index]
    
    test_data = D[test_index, :, :, :]
    test_labels = labels[test_index]

    number_tst = total_number - index # Number of testing images    

    learn_rate = (1. / (1. + 0.89* epoch))*initial_learn_rate

    train_loss = 0
    train_acc = 0

    test_loss = 0
    test_acc = 0

    # Testing images also need to fed in batches to avoid "Tensor out of memory error"
    # Hence loss and accuracy over test dataset is reported in steps

    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y, training: True, pkeep: 0.5, pkeep_conv: 0.8, lr: learn_rate})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0, pkeep_conv: 1.0, lr: learn_rate})
        train_acc = a
        train_loss = c   

        # Save model after each iteration
        save_path = saver.save(sess,"D:/model_10categories_data_aug")

        # Test calculation initialization
        acc = 0
        loss = 0
        
        # Line 569 - 578 => Testing loss # tf.reduce_mean is used hence there is no need to compute the average again
        for j in range(0, number_tst, batch_size):
            # start, stop, step
            batch_Xtst, batch_Ytst = get_next_batch(j, test_data, test_labels, batch_size)
            a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0, pkeep_conv: 1.0, lr: learn_rate})
            acc = acc + a
            loss = loss + c
            
        
        test_acc = (acc * batch_size)/ number_tst
        test_loss = (loss * batch_size)/ number_tst
        
        # Print model train loss and accuracy after each iteration
        print("Iteration",iters1,"Train Loss",train_loss,"Train Acc",train_acc) 
        print("Iteration",iters1,"Test Loss",test_loss,"Test Acc",test_acc) 
        print("\n")
        f.write("%i %f %f %f %f\n" % (iters1, train_loss, train_acc, test_loss, test_acc))
        iters1 = iters1 + 1 # update count of iterations 

f.close()        
#np.savez('CPSC_8810/plots/iterations_plots_4.npz',name1 = train_acc, name2 = train_loss, name3 = test_acc, name4 = test_loss, name5 = iters1)


