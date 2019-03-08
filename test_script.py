import os, sys
from PIL import Image
import numpy as np
import tensorflow as tf

ROWS = 224
COLS = 224
lr = 0.001
window = 5
#tf.set_random_seed(0)

for infile in sys.argv[1:]:
	img = Image.open(infile)
	img = img.resize((COLS, ROWS), Image.ANTIALIAS)

img = np.array(img)
# print(img)
# print(img.shape)
images = np.zeros(shape = (1, ROWS, COLS, 3))
# Detect if image is grayscale
if(len(img.shape) == 2):
    # Normalize the image
    temp = img;    
           
    temp = (temp - np.mean(temp, axis = (0,1)))/np.std(temp, axis = (0,1))                    
    # temp = temp/255

    # Copy grayscale into each channel seperately
    images[0, :, :, 0] = temp
    images[0, :, :, 1] = temp
    images[0, :, :, 2] = temp
else:
	i1 = img[:,:,0]
	i1 = (i1 - np.mean(i1, axis = (0, 1)))/np.std(i1, axis = (0, 1))

	i2 = img[:,:,1]
	i2 = (i2 - np.mean(i2, axis = (0, 1)))/np.std(i2, axis = (0, 1))

	i3 = img[:,:,2]
	i3 = (i3 - np.mean(i3, axis = (0, 1)))/np.std(i3, axis = (0, 1))

	img[:,:,0] = i1
	img[:,:,1] = i2
	img[:,:,2] = i3

	images[0,:,:,:] = img	

#print(images)
#print(images.shape)

# Reset Graph
tf.reset_default_graph()

# Create placeholders
X = tf. placeholder(tf.float32, [None, ROWS, COLS, 3])
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

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
	saver.restore(sess,"model_NB/model_BVNB")
	prediction = sess.run([Y_pred], feed_dict = {X: images, Y: np.reshape(1, newshape = (1)), pkeep: 1.0})
	#print('prediction', prediction)

pred = np.array(prediction)	
category = np.argmax(pred)
#print(category)

if(category == 0):
	print('Type : Non Bullying')

elif(category == 1):
	print('Type : Bullying')
	
	# Reset Graph
	tf.reset_default_graph()

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

	Y1_1 = tf.nn.conv2d(Y1_out, W1_1, strides = [1,1,1,1], padding = 'SAME') + B1_1 # Image Size => 112 x 112
	Y1_1_max = tf.nn.max_pool(Y1_1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	Y1_1_out = tf.nn.relu(Y1_1_max) # Image Size 56 x 56

	Y1_2 = tf.nn.conv2d(Y1_1_out, W1_2, strides = [1,1,1,1], padding = 'SAME') + B1_2 # Image Size => 56 x 56
	Y1_2_max = tf.nn.max_pool(Y1_2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')
	Y1_2_out = tf.nn.relu(Y1_2_max) # Image Size 28 x 28

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

	init = tf.global_variables_initializer()
	saver = tf.train.Saver()

	with tf.Session() as sess:
		saver.restore(sess,"model_new/model_9876")
		prediction = sess.run([Y_pred], feed_dict = {X: images, Y: np.reshape(1, newshape = (1)), pkeep: 1.0})
		#print('prediction', prediction)

	pred = np.array(prediction)	
	category = np.argmax(pred)
	#print(category)

	if(category == 1):
		print('Category : Gossiping')
	elif(category == 2):
		print('Category : Isolation')
	elif(category == 3):
		print('Category : Laughing')
	elif(category == 4):
		print('Category : Pulling Hair')
	elif(category == 5):
		print('Category : Punching')
	elif(category == 6):
		print('Category : Quarrel')
	elif(category == 7):
		print('Category : Slapping')
	elif(category == 8):
		print('Category : Stabbing')
	elif(category == 9):
		print('Category : Strangle')
	#elif(category == 0):
		#print('Non-bullying')