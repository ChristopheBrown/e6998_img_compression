import tensorflow as tf
import pandas as pd
import numpy as np

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import Input

import matplotlib.pyplot as plt

def padding(x, stride):
    if x % stride == 0:
        return x // stride
    else:
        return x // stride + 1

def init_hiddens(filters, size):
    
    data = np.random.normal(size=size)

    height = int(padding(data.shape[1], 2) / 2)
    width = int(padding(data.shape[2], 2) / 2)
    
    shape = (height, width, filters)
    hidden = tf.zeros(shape)
    cell = tf.zeros(shape)
        
    return (hidden, cell)

class RnnConv(Layer):
    
    # initialization of the layer, where layer attributes are created
    def __init__(self, filters, kernel_size, strides, index):
        
        super(RnnConv, self).__init__()
        
        # initialized hiddens dims must match the input tensor shape- which gets smaller with deeper layers
        h_w_scale_factor = 2**(index-1)
        h_w = int(32/h_w_scale_factor) #32 is the input dims, decreasing by factor of 2 for each layer
        
        self.kernel_size = kernel_size
        self.strides = strides
        self.filters = filters
        self.hidden, self.cell = init_hiddens(self.filters, size=(filters, h_w, h_w, 3))

    # building the layer, where the dimensions of the layer are assembled once they are known at composition time
    def build(self,input_shape):
        
        # conv_innput was removed because the input convolution is only done at the beginning and not repeated for rnn_conv
        self.conv_hidden = Conv2D(filters=self.filters,  #multiple by 4 because it needs to be split into 4 gates?
                                  kernel_size=self.kernel_size, 
                                  strides=self.strides, 
                                  activation='relu', 
                                  padding="same")

        self.dense_sigmoid = Dense(self.filters, activation="sigmoid")
        self.dense_tanh = Dense(self.filters, activation="tanh")
    
    # calling the layer, where the arithmetic and forward computations are performed
    def call(self, inputs):
        
        self.conv2 = self.conv_hidden(inputs)
        
#         in_gate, forget_gate, out_gate, cell_state = tf.split(inputs + self.conv2, 4, axis=-1)
        in_gate, forget_gate, out_gate, cell_state = tf.split(self.conv2, 4, axis=-1)
    
        self.in_gate = self.dense_sigmoid(in_gate)
        self.forget_gate = self.dense_sigmoid(forget_gate)
        self.out_gate = self.dense_sigmoid(out_gate)
        self.cell_state = self.dense_tanh(cell_state)

        
#         print(f'cell shape: {self.cell.shape}')
        # needed to add a tf.squeeze() to get rid of an arbitrary 5th dim that was added
        new_cell_p1 = tf.multiply(self.forget_gate, tf.expand_dims(self.cell, axis=0))
        if (len(new_cell_p1.shape) == 5):
            new_cell_p1 = tf.squeeze(new_cell_p1, axis=0)
#         print(f' new cell shape: {new_cell_p1.shape}')
#         print(f' squeezed shape: {tf.squeeze(new_cell_p1,axis=0).shape}')

        
        new_cell_p2 = tf.multiply(self.in_gate, self.cell_state)
        self.new_cell = new_cell_p1 + new_cell_p2
        self.new_hidden = tf.multiply(self.out_gate, Dense(1, activation="tanh")(self.new_cell))
        
        self.cell = self.new_cell # should this update the state of the LSTM cell in this layer (and not propagate forward)
        self.hidden = self.new_hidden
        
#         return self.new_hidden , self.new_cell
        return self.new_cell
        # had to comment the second return because only one tensor can be returned from the layer
