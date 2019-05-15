# Used libraries

import tensorflow as tf 
from MultiLayerPerceptron import MLP
from CustomFunctions import *
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from tensorflow.keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():

    # Load the dataset
    (X_train, y_train), (X_test, y_test) = load_dataset()

    # Normalizes the dataset
    X_train = normalize_dataset(X_train)
    X_test  = normalize_dataset(X_test)

    # Create the Neural Network Structure

    # 1. Define the number of Hidden Layers and number of Neurons at each layer 
    neurons_per_layer = [16] # 1 layer, 20 neurons

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
    #   - 'relu'
    #   - 'softmax'
    #   - 'sigmoid'
    out_layer_activation = 'sigmoid'

    # 5. Define the learning rate
    #   - Variate the learning rates!
    learning_rate = 0.01

    # 6. Define the optimizer
    #  Options:
    #   - 'adam'
    #   - 'sgd'
    #   - SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    optimizer = SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

    # 7. Define the loss function
    # Options: 
    #   - 'categorical_crossentropy'
    #   - 'mean_squared_error'
    #   -  sqr_error
    loss = 'categorical_crossentropy'

    # 8. Number of epochs
    epochs = 20

    ## Learn and Predict Model

    # 1. Construct the MLP
    mlp = MLP(neurons_per_layer,
              output_classes,
              activation_functions,
              out_layer_activation,
              optimizer,
              loss,
              learning_rate)

    # 2. Fit the model
    mlp.learn(X_train, to_categorical(y_train), epochs=epochs)

    # 3. Show recognized patterns
    for output_class in [1, 3, 7]:
        pattern_recognized, pattern_rejected, total_pattern = mlp.get_pattern_to_classify_as(output_class)
        show_image(pattern_recognized, label = 'pattern recognized for class {}'.format(output_class),
                   new_figure=True, block=False)

        #show_image(pattern_rejected, label = 'pattern rejected for class {}'.format(output_class),
                   #new_figure=True, block=False)

        #show_image(total_pattern, label = 'Remainder pattern for class {}'.format(output_class),
                   #new_figure=True, block=False)

    #plt.show()

    # 4. Evaluate the model
    test_classes = mlp.model.predict_classes(X_test)
    test_outputs = mlp.predict(X_test)

    #print ('Accuracy on Training Set:', accuracy_score(y_train, mlp.model.predict_classes(X_train)))
    #print ('F1-Score on Training Set:', f1_score(y_train, mlp.model.predict_classes(X_train), average='macro'))
    #print ('Confusion Matrix on Training Set:')
    #print (confusion_matrix(y_train, mlp.model.predict_classes(X_train)))

    print ('Accuracy on Test Set:', accuracy_score(y_test, mlp.model.predict_classes(X_test)))
    print ('F1-Score on Test Set:', f1_score(y_test, mlp.model.predict_classes(X_test), average='macro'))
    print ('Classification Report on Test Set:')
    plot_confusion_matrix(confusion_matrix(y_test, mlp.model.predict_classes(X_test)))

    # 5. Show Evaluation Plots
    show_evaluation(mlp)

def load_dataset():
    mnist = tf.keras.datasets.mnist
    return mnist.load_data()

def normalize_dataset(X):
    return tf.keras.utils.normalize(X)


if __name__ == "__main__":
    main()