# Used libraries

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt

# Multilayer Perceptron Model
class MLP:
    def __init__(self, 
                neurons_per_layer,
                output_classes,
                activation_functions,
                out_layer_activation,
                optimizer,
                loss,
                learning_rate):

        # Attributes
        self.neurons_per_layer = neurons_per_layer
        self.activation_functions = activation_functions
        self.out_layer_activation = out_layer_activation
        self.optimizer = optimizer
        self.loss = loss
        self.learning_rate = learning_rate

        # Feed-Forward Neural Network Model
        self.model = tf.keras.models.Sequential()

        # Flatten the 28x28 images provided
        self.model.add(tf.keras.layers.Flatten())

        # Hidden Layers
        for neurons, act in zip(neurons_per_layer, activation_functions):
            self.model.add(tf.keras.layers.Dense(neurons, activation=act))

        # Output Layer
        self.model.add(tf.keras.layers.Dense(output_classes,
            activation=out_layer_activation))

        # Build Model
        self.model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    
    def learn(self, X, y, epochs):
        self.history = self.model.fit(X, y, epochs=epochs)
        print (self.history.history.keys())

    def predict(self, X_new, batch_size=100):
        predictions = self.model.predict(X_new, batch_size=batch_size)
        return predictions

        #layer, neuron integers from 0...N-1
    def get_weigths_from_neuron_at_layer(self, neuron, layer):
        weights = []
        for W_array in self.model.layers[layer].get_weights()[0]: #0 = weights, 1 = bias
            weights.append(W_array[neuron])
        return(np.array(weights))

    def get_binary_matrix_from_neuron(self, neuron):
        weights = self.get_weigths_from_neuron_at_layer(neuron, -2) # hidden layer closer to output layer
        weights[weights<0] = 0 # white
        #weights[weights>0] = 1 # black
        weights = weights/np.amax(weights) # values between 0 and 1
        img = weights.reshape((28, 28))
        return img

    def plot_first_layer_weight_matrix_from_neuron(self, neuron, label='', new_figure=False, block=True):
        img = self.get_binary_matrix_from_neuron(neuron)
        if(new_figure):
            plt.figure() # new figure to accomodate every loop image 
        plt.imshow(img, cmap="Greys")
        plt.title(label)
        plt.show(block = block)

    def get_pattern_to_classify_as(self, img_class):
        output_layer_w = self.get_weigths_from_neuron_at_layer(neuron=img_class, layer=-1)# output layer
        sorted_indices = np.argsort(output_layer_w)[::-1] # descending order
        pattern_recognized = np.zeros((28,28)) # sum of positive weights 
        pattern_rejected = np.zeros((28,28)) # sum of negative weights
        total_pattern = np.zeros((28,28)) # sum of all weights (recognized - rejected)
        for i in sorted_indices:
            np.add(total_pattern, output_layer_w[i]*self.get_binary_matrix_from_neuron(i), out=total_pattern)
            if(output_layer_w[i]>0): # this pattern is accepted as to activate output from this class 
                img = output_layer_w[i]*self.get_binary_matrix_from_neuron(i)
                np.add(pattern_recognized, img, out=pattern_recognized)
            else: # this pattern is rejected as activation output from this class
                img = -output_layer_w[i]*self.get_binary_matrix_from_neuron(i)
                np.add(pattern_rejected, img, out=pattern_rejected)
        pattern_recognized /= np.amax(pattern_recognized)
        pattern_rejected /= np.amax(pattern_rejected)
        total_pattern/=np.amax(total_pattern)
        total_pattern[total_pattern<0] = 0
        return(pattern_recognized, pattern_rejected, total_pattern)