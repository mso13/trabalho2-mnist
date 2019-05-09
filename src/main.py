# Used libraries

import tensorflow as tf 
from MultiLayerPerceptron import MLP
from CustomFunctions import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Function to plot an image
def show_image(img, label,new_figure=False,block=True):
    if(new_figure):
        plt.figure() # new figure to accomodate every loop image 
    plt.imshow(img, cmap="Greys")
    plt.title(label)
    plt.show(block = block)

def main():

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = load_dataset()

    # Normalizes the dataset
    X_train = normalize_dataset(X_train)
    X_test  = normalize_dataset(X_test)

    # Create the Neural Network Structure

    # 1. Define the number of Hidden Layers and number of Neurons at each layer 
    neurons_per_layer = [20] # 1 layer, 20 neurons

    # 2. Define the number of output classes
    output_classes = 10

    # 3. Define the activation function on hidden layers
    # Options:
    #   - 'relu'
    #   - 'tanh'
    #   - 'sigmoid'
    activation_functions = ['sigmoid']

    # 4. Define the activation function on output layer
    # Options:
    #   - 'softmax'
    #   - 'sigmoid'
    out_layer_activation = 'softmax'

    # 5. Define the learning rate
    #   - Variate the learning rates!
    learning_rate = 0.01

    # 6. Define the optimizer
    #  Options:
    #   - 'adam'
    #   - SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = SGD(lr=learning_rate, momentum=0.0, decay=0.0, nesterov=False)

    # 7. Define the loss function
    # Options: 
    #   - 'categorical_crossentropy'
    #   - 'mean_squared_error'
    #   -  sqr_error
    loss = 'categorical_crossentropy'



    # Learn and Predict Model

    # 1. Construct the MLP
    mlp = MLP(neurons_per_layer,
              output_classes,
              activation_functions,
              out_layer_activation,
              optimizer,
              loss,
              learning_rate)

    # 2. Fit the model
    mlp.learn(X_train, to_categorical(y_train), epochs=5)
    # 3. Show recognized patterns
    for output_class in range(10):
        pattern_recognized, pattern_rejected, total_pattern = mlp.get_pattern_to_classify_as(output_class)
        show_image(pattern_recognized, label = 'pattern recognized for class {}'.format(output_class),
                   new_figure=True, block=False)
        #show_image(pattern_rejected, label = 'pattern rejected for class {}'.format(output_class),
                   #new_figure=True, block=False)
        show_image(total_pattern, label = 'Remainder pattern for class {}'.format(output_class),
                   new_figure=True, block=False)
    plt.show()

    # 4. Evaluate the model
    test_classes = mlp.model.predict_classes(X_test)
    test_outputs = mlp.predict(X_test)

    print('Accuracy and CM on Training Set:')
    print(accuracy_score(y_train, mlp.model.predict_classes(X_train)))
    print(confusion_matrix(y_train, mlp.model.predict_classes(X_train)))

    print('Accuracy and CM on Test Set:')
    print(accuracy_score(y_test, mlp.model.predict_classes(X_test)))
    print(confusion_matrix(y_test, mlp.model.predict_classes(X_test)))

    # Previous attempt

    # mlp = MLP([60, 10], 0.01)
    # print('non categorical shape and first sample')
    # print(y_train.shape)
    # print(y_train[0])
    # print('categorical shape and first sample')
    # print(to_categorical(y_train).shape)
    # print(to_categorical(y_train)[0])

    # # Converting y_train to categorical will transform the outputs
    # # as a one hot variable (1 for the desired class and 0 for the others)
    # # allowing the network to train each output neuron.
    
    # mlp.learn(x_train, to_categorical(y_train), epochs=10)
    # test_classes  = mlp.model.predict_classes(x_test)
    # test_outputs = mlp.predict(x_test)


def load_dataset():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()

def normalize_dataset(X):
    return tf.keras.utils.normalize(X)


if __name__ == "__main__":
    main()