########################################## Funções de Custo/Ativação ############################################################
import tensorflow as tf
from keras import backend as K

def sqr_error(y_true, y_pred): # it simplifies gradient descend
    return  K.sum(K.square(y_pred - y_true), axis=None)

def custom_activation(x):
#sigmoide quadrada para reduzir a ativacao do neuronio caso as entradas sejam inibitorias (negativas ou proximas a 0)
    return K.sigmoid(x)*K.sigmoid(x)


########################################## Visualização dos Dados ###############################################################

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels

def plot_confusion_matrix(cm, normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Greens):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    classes = [str(x) for x in range(10)]

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


# Function to plot an image
def show_image(img, label,new_figure=False,block=True):
    if(new_figure):
        plt.figure() # new figure to accomodate every loop image 
    plt.imshow(img, cmap="Greys")
    plt.title(label)
    plt.show(block = block)

def show_evaluation(network, block=True):

    history = network.model.history

    # Summarize history for Accuracy
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy\nLR={}, Hidden={}, Activations={}\nOutput={}, Loss={}'
                .format(network.learning_rate,
                 network.neurons_per_layer,
                 network.activation_functions,
                 network.out_layer_activation,
                 network.loss))
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show(block = block)

    # Summarize history for Loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss\nLR={}, Hidden={}, Activations={}\nOutput={}, Loss={}'
                .format(network.learning_rate,
                 network.neurons_per_layer,
                 network.activation_functions,
                 network.out_layer_activation,
                 network.loss))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show(block = block)