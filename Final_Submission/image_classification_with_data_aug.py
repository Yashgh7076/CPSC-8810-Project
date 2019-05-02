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
lr = 0.0001

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

L = np.array([l1, L2, L3, L4, L5, L6, L7, L8, L9], dtype = np.int32)

# Read Standford 40 Images
filename_10 = 'stanford/'
a10 = sorted(os.listdir(filename_10))
l10 = len(a10)
number = l10

total_number = (4*L9) + number

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

#---------------Data Augmentation------------------#
D[L9:(2*L9), :, :, :] = flip_images(D[0:L9, :, :, :]) # Flip images horizontally
D[(2*L9):(3*L9),:,:,:] = jitter(D[0:L9, :, :, :], 0.25) # Brightness jitter

D[(3*L9):(4*L9),:,:,:] = flip_images(D[0:L9, :, :, :]) # Flip + Additive Gaussian noise
D[(3*L9):(4*L9),:,:,:] = add_noise(D[(3*L9):(4*L9), :, :, :], 0, 0.25)

# Augment labels
augment_labels(labels, L9, L) # augment labels for flipped images

augment_labels(labels, (2*L9), L) # augment labels for jitter images

augment_labels(labels, (3*L9), L) # augment labels for flipped + noise images

#---Read Stanford 40 images---#
D[(4*L9):total_number, :, :, :] = read_nonbullying(filename_10, number)
labels[(4*L9):total_number] = 0

D = standard_scaler(D)

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

# Regularization Part
reg = tf.nn.l2_loss(W1) + tf.nn.l2_loss(W1_1) + tf.nn.l2_loss(W1_2) 
fc = tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)
o = tf.nn.l2_loss(W4)

final_reg = tf.reduce_mean(reg + fc + o)
cross_entropy = cross_entropy + 0.00001 * final_reg


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

epochs = 250

train_loss = np.zeros(shape=(epochs))
train_acc = np.zeros(shape=(epochs))

test_loss = np.zeros(shape=(epochs))
test_acc = np.zeros(shape=(epochs))


f = open('model_simple1.txt','w')
f.write("Iteration Training_Loss Training_Acc Test_Loss Test_Acc \n")

for epoch in range(epochs):
    batch = 1     
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

    number_tst = total_number - index
    batch_tst = 1
    # Number of testing images    

    # Testing images also need to fed in batches to avoid "Tensor out of memory error"
    # Hence loss and accuracy over test dataset is reported in steps

    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y, pkeep: 0.4})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0})
        train_acc[epoch] = train_acc[epoch] + a
        train_loss[epoch] = train_loss[epoch] + c        
        batch = batch + 1
        
    train_loss[epoch] = train_loss[epoch]/batch
    train_acc[epoch] = train_acc[epoch]/batch
    save_path = saver.save(sess,"F:/model_10categories")
    
    #testdata = {X: test_data, Y:test_labels, pkeep: 1.0}
    #a,c = sess.run([accuracy,cross_entropy], feed_dict = testdata)
    #test_acc = a
    #test_loss = c

    #batch_tst = 1   

    for j in range(0, number_tst, batch_size):
        #start, stop, step
        batch_Xtst, batch_Ytst = get_next_batch(j, test_data, test_labels, batch_size)
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})
        test_acc[epoch] = test_acc[epoch] + a
        test_loss[epoch] = test_loss[epoch] + c        
    
    

    test_loss[epoch] = (test_loss[epoch] * batch_size)/number_tst
    test_acc[epoch] = (test_acc[epoch] * batch_size)/number_tst

    print("Epoch",epoch ,"Train Loss",train_loss[epoch],"Train Acc",train_acc[epoch])
    print("Epoch",epoch,"Test Loss",test_loss[epoch],"Test Acc",test_acc[epoch])
    print("\n")
    f.write("%i %f %f %f %f\n" % (epoch, train_loss[epoch], train_acc[epoch], test_loss[epoch], test_acc[epoch]))
    #saver.save(session, 'weights_model')
    epoch = epoch + 1 

#np.savez('CPSC_8810/plots/10categoriesnew_plots.npz',name1 = train_acc, name2 = train_loss, name3 = test_acc, name4 = test_loss)
