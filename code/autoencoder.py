
import numpy as np
import tensorflow  as tf 
import os
from random import shuffle
np.set_printoptions(threshold=np.nan)

class CNNAutoencoder(object):

    def __init__(self, img_height, img_width, img_depth, learning_rate=1e-3, name=None):
        '''
        Create an CNN Autoencoder instance
        @param 
            img_height: each image's height
            img_width: each image's width
            learning_rate: the learning rate for training, default to be 0.001
            name: the autoencoder's name

        '''
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.learning_rate = learning_rate
        self.name = name if name else 'CNNAutoencoder'

        self.step = 0

    #build the model by creating tensorflow graph of each component of the model
    def build_graph(self):

        self._build_model()
        self._build_train_op()

    def _build_train_op(self):
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
    
    def _build_model(self):
        
        self.X = tf.placeholder(dtype=tf.float32, shape=(None, self.img_height, self.img_width, self.img_depth))
        self.y = tf.placeholder(dtype = tf.float32,shape=(None,self.img_height,self.img_width,self.img_depth))
        
        # Encoder, three convolution layers, three max pooling layers
        self.x = tf.layers.conv2d(self.X, filters=16, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv1' % self.name)
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_pool1' % self.name)
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv2' % self.name)
        self.x = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_pool2' % self.name)
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv3' % self.name)
        self.encoded = tf.layers.max_pooling2d(self.x, pool_size=(2, 2), strides=(2, 2), padding='SAME', name='%s_encoded' % self.name)

        # Decoder, three convolution layers, three deconvolution layers and one reconstruction layer
        self.x = tf.layers.conv2d(self.encoded, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv4' % self.name)
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv4' % self.name)
        self.x = tf.layers.conv2d(self.x, filters=8, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_conv5' % self.name)
        self.x = tf.layers.conv2d_transpose(self.x, filters=8, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv5' % self.name)
        self.x = tf.layers.conv2d(self.x, filters=16, kernel_size=(3, 3), activation=tf.nn.relu,padding='SAME', name='%s_conv6' % self.name)
        self.x = tf.layers.conv2d_transpose(self.x, filters=16, kernel_size=(2, 2), strides=(2, 2), name='%s_deconv6' % self.name)
        self.recon = tf.layers.conv2d(self.x, filters=1, kernel_size=(3, 3), activation=tf.nn.relu, padding='SAME', name='%s_recon' % self.name)

        #save self.X and self.recon for prediction use
        tf.get_default_graph().add_to_collection("Input",self.X)
        tf.get_default_graph().add_to_collection("Output",self.recon)
        
        # Use mean-square-error as loss
        self.loss = tf.reduce_mean(tf.square(self.y - self.recon))

        self.loss_summary = tf.summary.scalar("Loss", self.loss)

    
    def get_reconstructed_img(self, sess, X):
        '''
        Get stripped images of the input
        @param
            sess: tensorflow session
            X: the test image set
        
        @return
            the stripped images
        '''

        feed_dict = {
            self.X: X
        }

        return sess.run(self.recon, feed_dict=feed_dict)

