# Here we are going to define the functions that help loading the data and other trivial stuff
from __future__ import print_function

import numpy as np                 # to use numpy arrays
import tensorflow as tf            # to specify and run computation graphs
import tensorflow_datasets as tfds # to load training data
import matplotlib.pyplot as plt    # to visualize data and draw plots

def load_data():
    # The first 90% of the training data as a dic
    # Use this data for the training loop
    train = tfds.load('mnist', split='train[:90%]',batch_size=-1)

    # And the last 10%, we'll hold out as the validation set as a dic
    val = tfds.load('mnist', split='train[-10%:]', batch_size=-1)
    
    train_x = tf.reshape(tf.cast(train['image'], tf.float32), [-1, 784])
    
    train_y = train['label']
    
    val_x = tf.reshape(tf.cast(val['image'], tf.float32), [-1, 784]) 
    
    val_y = val['label']
    
    return train_x, train_y, val_x, val_y

def show_history(history,key,num):
    fig = plt.figure()
    plt.plot(history.history[key])
    plt.title(key)
    plt.ylabel(key)
    plt.xlabel('No. epoch')
    plt.show()
    fig.savefig("./"+ str(num) + '_' + key + ".png",  bbox_inches='tight')
    np.savetxt(str(num) + '_' + key+".txt", history.history[key]);
    
    