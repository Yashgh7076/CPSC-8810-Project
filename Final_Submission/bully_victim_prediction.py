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
#ROWS = 224
#COLS = 224
newsize = 224
window = 9
tf.set_random_seed(0)
lr = 0.0001

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
    a = sorted(os.listdir(filename)) # Sorted list of files
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
    
    return(l)            

def flip_images(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))
    
    for i in range(N):
        for j in range(3):
            temp = dataset[i,:,:,j]
            rot  = np.fliplr(temp) # Actual Flip LR operation
            images[i,:,:,j] = rot
            
    return(images)
    
def add_noise(dataset, mean, std_dev):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))    
    noise = np.random.normal(mean, std_dev, (newsize, newsize, 3))    
    images = dataset + noise   
    
    return(images)
    
def jitter(dataset, brightness):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))    
    bright = np.ones((newsize, newsize, 3)) + np.random.uniform(-brightness, brightness, (newsize, newsize, 3))    
    images = dataset * bright
    
    return(images)

def get_next_batch(index, dataset, classes, batch_size):
    
    X_batch = np.array(dataset[index:index + batch_size,:,:,:], dtype = np.float32)
    
    Y_batch = np.array(classes[index:index + batch_size], dtype = np.int32)
    
    return(X_batch,Y_batch) 

def augment_labels(labels, previous, labels_temp, total_images):
    
    labels[previous: previous + total_images] = labels_temp 


def standard_scaler(dataset):

    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3)) 
    img = np.zeros(shape = (newsize, newsize))

    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            img_scl = (img - np.mean(img, axis = (0, 1)))/np.std(img, axis = (0, 1))
            images[i, :, :, c] = img_scl
    
    return(images)

def translate(dataset, shift_cols, shift_rows):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3)) 
    img = np.zeros(shape = (newsize, newsize))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            M = np.float32([[1,0,shift_cols],[0,1,shift_rows]])
            images[i,:,:,c] = cv.warpAffine(img,M,(newsize, newsize))
    
    return(images)
    
def rotate_90(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))
    img = np.zeros(shape = (newsize, newsize))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            M = cv.getRotationMatrix2D(((newsize - 1)/2.0,(newsize - 1)/2.0),90,1)
            images[i,:,:,c] = cv.warpAffine(img,M,(newsize, newsize))
    
    return(images)


def affine_transform(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))
    img = np.zeros(shape = (newsize, newsize))
    # 3 points in original image -> rotate 3 points in destination image
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            pts1 = np.float32([[10,10],[200,10],[10,200]])
            pts2 = np.float32([[10,100],[200,10],[100,200]])
            M = cv.getAffineTransform(pts1,pts2)
            images[i, :, :, c] = cv.warpAffine(img,M,(newsize, newsize))
            
    return(images)
           
def perspective_transform(dataset):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))
    img = np.zeros(shape = (newsize, newsize))
    
    # 4 points of perspective transform
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            pts1 = np.float32([[30,30],[150,30],[30,190],[190,190]])
            pts2 = np.float32([[0,0],[newsize,0],[0,newsize],[newsize, newsize]])
            M = cv.getPerspectiveTransform(pts1,pts2)
            images[i, :, :, c] = cv.warpPerspective(img,M,(newsize, newsize))
    
    return(images)

def avg_smooth(dataset, region):
    
    N = dataset.shape[0]
    images = np.zeros(shape = (N, newsize, newsize, 3))
    img = np.zeros(shape = (newsize, newsize))
    
    for i in range(N):
        for c in range(3):
            img = dataset[i, :, :, c]
            images[i, :, :, c] = cv.blur(img,(region,region))
            
    return(images)

# Read files and labels
filename_1 = 'Labelling/'

with open ('DL_labels.json') as json_file:
    data = json.load(json_file)

'''
D = np.zeros(shape = (5000, newsize, newsize, 3))
labels = np.zeros(shape = (5000))

L10 = read_from_folder(filename_1, data, D, labels) 

D_temp = np.zeros(shape = (L10, newsize, newsize, 3))
labels_temp = np.zeros(shape = L10) 

D_temp = D[0:L10, :, :, :] # Copy images + labels
labels_temp = labels[0:L10]

D = np.zeros(shape = (total_number, ROWS, COLS, 3)) 
labels = np.zeros(shape = (total_number)) 


del D # Free images + labels temporarily
del labels
'''


D = np.zeros(shape = (5000, newsize, newsize, 3))
labels = np.zeros(shape = (5000))

L10 = read_from_folder(filename_1, data, D, labels) 

D_temp = np.zeros(shape = (L10, newsize, newsize, 3))
labels_temp = np.zeros(shape = L10) 

D_temp = D[0:L10, :, :, :] # Copy images + labels
labels_temp = labels[0:L10]



del D # Free images + labels temporarily
del labels

total_number = 10*L10

D = np.zeros(shape = (total_number, newsize, newsize, 3))
labels = np.zeros(shape = (total_number))

D[0:L10,:,:,:] = D_temp
labels[0:L10] = labels_temp






del D_temp

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
augment_labels(labels, L10, labels_temp, L10) # augment labels for flipped images

augment_labels(labels, (2*L10), labels_temp, L10) # augment labels for jitter images

augment_labels(labels, (3*L10), labels_temp, L10) # augment labels for flipped + noise images

augment_labels(labels, (4*L10), labels_temp, L10)

augment_labels(labels, (5*L10), labels_temp, L10)

augment_labels(labels, (6*L10), labels_temp, L10)

augment_labels(labels, (7*L10), labels_temp, L10)

augment_labels(labels, (8*L10), labels_temp, L10)

augment_labels(labels, (9*L10), labels_temp, L10)
#--------------Model Starts Here--------------------#
# Create placeholders
# Create placeholders
X = tf. placeholder(tf.float32, [None, newsize, newsize, 3])
Y = tf.placeholder(tf.int32,[None])
depth = 2 # The number of classes
Y_onehot = tf.one_hot(Y,depth)
# lr = tf.placeholder(tf.float32)
pkeep = tf.placeholder(tf.float32)

# Specify parameters of the NN model & training 
depth_1 = 10 # CNN Layer 1 o/p channels
depth_2 = 8
depth_3 = 6
fc_1 = 200
fc_2 = 100

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
W4 = tf.Variable(tf.truncated_normal(shape = [fc_2, 2], stddev = 0.1))
B4 = tf.Variable(tf.constant(0.1, tf.float32, shape = [2]))

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

batch_size = 8  

epochs = 10

initial_learn_rate = 0.0001

batch_acc = np.zeros(epochs)
batch_loss = np.zeros(epochs)
test_loss = np.zeros(epochs)
test_acc = np.zeros(epochs)

f = open('model_6.txt','w')
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

    # Testing images also need to fed in batches to avoid "Tensor out of memory error"
    # Hence loss and accuracy over test dataset is reported in steps

    batch_number = 0   

    for i in range(0, index, batch_size):
        #start, stop, step
        batch_X, batch_Y = get_next_batch(i, train_data, train_labels, batch_size)
        sess.run(train_step, feed_dict = {X: batch_X, Y: batch_Y,pkeep: 0.5})
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_X, Y: batch_Y, pkeep: 1.0})
        train_acc = a
        train_loss = c   

        # Save model after each iteration
        save_path = saver.save(sess,"X:/model_2")

        # Write output to file after each iteration
        f.write("%i %f %f\n" % (iters1, train_loss, train_acc))
        iters1 = iters1 + 1 # update count of iterations

        batch_acc[epoch] = batch_acc[epoch] + train_acc
        batch_loss[epoch] = batch_loss[epoch] + train_loss
        batch_number = batch_number + 1

    batch_acc[epoch] = batch_acc[epoch]/batch_number
    batch_loss[epoch] = batch_loss[epoch]/batch_number

    # Test calculation initialization
    acc = 0
    loss = 0
    # Line 569 - 578 => Testing loss # tf.reduce_mean is used hence there is no need to compute the average again
    for j in range(0, number_tst, batch_size):
        # start, stop, step
        batch_Xtst, batch_Ytst = get_next_batch(j, test_data, test_labels, batch_size)
        a,c = sess.run([accuracy,cross_entropy], feed_dict = {X: batch_Xtst, Y: batch_Ytst, pkeep: 1.0})
        acc = acc + a
        loss = loss + c
       
    test_acc[epoch] = (acc * batch_size)/ number_tst
    test_loss[epoch] = (loss * batch_size)/ number_tst
        
    # Print model train loss and accuracy after each iteration
    print("Epoch",epoch,"Train Loss",batch_loss[epoch],"Train Acc",batch_acc[epoch]) 
    print("Epoch",epoch,"Test Loss",test_loss[epoch],"Test Acc",test_acc[epoch]) 
    print("\n")         

f.close()        
np.savez('plots/resnet_4_128.npz',name1 = batch_acc, name2 = batch_loss, name3 = test_acc, name4 = test_loss)