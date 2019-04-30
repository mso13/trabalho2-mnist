from imports import *
from custom_functions import *
from keras.optimizers import SGD
from keras.models import Sequential # to avoid mixing keras and tensorflow.keras 
from keras.layers import Dense, Flatten # to avoid mixing keras and tensorflow.keras
class GNN: # generic neural network model
    def __init__(self, neurons_per_layers, activations, optimizer ,cost_fun):
        self.model = Sequential()
        for neurons_per_layer, a_fun in zip(neurons_per_layers, activations):
            self.model.add(Dense(neurons_per_layer, activation=a_fun))
        self.model.compile(optimizer= optimizer,
                           loss=cost_fun,
                           metrics=['accuracy'])
    def learn(self, inputs, desired_outputs, epochs):
        self.model.fit(inputs, desired_outputs, epochs=epochs)

    def predict(self, inputs, batch_size=100):
        return(self.model.predict(inputs, batch_size=batch_size))

class MLP(GNN):
    def __init__(self, neurons_per_layers, LR):
        n_layers = len(neurons_per_layers)
        activations = [K.sigmoid]*n_layers
        cost = mean_sqr_error
        optimizer = SGD(lr=LR, decay=0.0, momentum=0.0, nesterov=False)
        GNN.__init__(self, neurons_per_layers, activations,optimizer, cost)



