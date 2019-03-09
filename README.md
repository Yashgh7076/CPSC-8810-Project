# CPSC-8810-Project
Deep Learning Project
# Contributors:
Yadnyesh Y. Luktuke and Eshaa D. Sood

# Objective:
1. Devise a model for classifying an image as bullying or not-bullying.
2. Also, train the model to differentiate between the 9 categories of bullying i.e gossiping, isolation, laughing, pulling hair, punching,    lapping, stabbing, strangling and quarelling.

# Network Architecture of Model:
The model consists of 3 convolution layers and 2 fully connected layers. The architecture of the model is as follows:

Convolutional Layer 1: The input image is fed into it, the filter size is 5*5, the number of channels is 3 and a total of 10 filters are used.

Convolutional Layer 2: The output of layer 1 is fed into it, filter size is 3*3, the number of channels is 10 and there are a total of 8 filters used.

Convolutional Layer 3: The output of layer 2 is fed into it, filter size is 3*3, the number of channels is 8 and there are a total of 6 filters used.

Fully Connected Layer 1: The output of layer 3 is fed into it, 28*28*6 is the number of inputs and it produces 200 outputs.

Fully Connected layer 2: The output of fully connected layer 1 is fed into it, the number of inputs is 200 and it produces 200 outputs.

The output of the fully connected layer 2 is then fed into the softmax activation function that gives the 10 probabilities.

The figure below is an accurate depiction of the architecture described above
![Architecture](https://user-images.githubusercontent.com/36894500/54064563-063fd280-41db-11e9-8217-001d7de46ab0.png)



# Requirements:
Python 3.6.2

GPU

Cuda-toolkit 9.0.176

CUDNN 9.0 v7

Anaconda 3.5.0.1

## Libraries:

Tensorflow_GPU-1.5.0

Matplotlib

Scipy

Pillow

Imageio

Numpy



# References:
Google Cloud Platform,
``Tensorflow and deep learning, without a PhD'', available at 
{"https://codelabs.developers.google.com/codelabs/cloud-tensorflow-mnist/#0"}

K.Simoyan, A.Zisserman,
``Very Deep Convolutional Networks for Large-Scale Image Recognition'', available at 
{"http://arxiv.org/abs/1409.1556"}, Online; Accessed \date{Feb 28, 2019}

B. Yao, X. Jiang, A. Khosla, A.L. Lin, L.J. Guibas, and L. Fei-Fei,
``Human Action Recognition by Learning Bases of Action Attributes and Parts'', International Conference on Computer Vision (ICCV), Barcelona, Spain. November 6-13, 2011

S. van der Walt, S. Chris Colbert and G. Varoquaux,
``The NumPy Array: A Structure for Efficient Numerical Computation'', Computing in Science \& Engineering, 13, 22-30 (2011)

A.Clark and contributors, F.Lundh and contributors,
``Pillow and Python Imaging Library (PIL)'', available at \url{https://pillow.readthedocs.io/en/stable/index.html}, Online; Accessed \date{Feb 28, 2019}

S.Theodoridis, K. Koutroumbas,
``Pattern Recognition, Fourth Edition'', Academic Press, Inc. 2008

A. Ng,
``Mini Batch Gradient Descent'', available at 
\url{"https://www.youtube.com/watch?v=4qJaSmvhxi8"}, Online; Accessed \date{Jan 20, 2019}
