from imports import *
import os
from neural_networks import GNN
from neural_networks import MLP
from keras.utils import to_categorical 
from sklearn.metrics import accuracy_score, confusion_matrix
# referencia para adicionar funcoes de ativacao personalizadas: 
#https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
#tentar conda install tensorflow-gpu caso esteja usando anaconda e ocorra erro de dll
#tf.enable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#def custom_activation(x):
#sigmoide quadrada para reduzir a ativacao do neuronio caso as entradas sejam inibitorias (negativas ou proximas a 0)
#    return K.sigmoid(x)*K.sigmoid(x)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train, x_test = x_train.reshape(x_train.shape[0], 28*28), x_test.reshape(x_test.shape[0], 28*28)

#print(type(x_train))
#print(type(x_test))
#print(x_train[0,:,:])
#print(x_train[0,:,:].shape)
#print(x_test[0,:,:])
#print(x_test[0,:,:].shape)
#print(x_train.shape)
#print(x_test.shape)
#print(y_train.shape)
#print(y_test.shape)

mlp = MLP([60, 10], 0.01)
print('non categorical shape and first sample')
print(y_train.shape)
print(y_train[0])
print('categorical shape and first sample')
print(to_categorical(y_train).shape)
print(to_categorical(y_train)[0])
#converting y_train to categorical will transform the outputs as a one hot variable 
#(1 for the desired class and 0 for the others) allowing the network to train each output neuron
mlp.learn(x_train, to_categorical(y_train), epochs=10)
test_classes  = mlp.model.predict_classes(x_test)
test_outputs = mlp.predict(x_test)

print(test_classes[0])
print(test_outputs[0])

print('Acuracia e cm no treino:')
print(accuracy_score(y_train, mlp.model.predict_classes(x_train)))
print(confusion_matrix(y_train, mlp.model.predict_classes(x_train)))

print('Acuracia e cm no teste:')
print(accuracy_score(y_test, mlp.model.predict_classes(x_test)))
print(confusion_matrix(y_test, mlp.model.predict_classes(x_test)))

#model = tf.keras.models.Sequential([
#  tf.keras.layers.Flatten(input_shape=(28, 28)),
#  tf.keras.layers.Dense(60, activation=custom_activation),
#  tf.keras.layers.Dense(10, activation='softmax')
#])
#model.compile(optimizer='adam',
#              loss='sparse_categorical_crossentropy',
#              metrics=['accuracy'])

#model.fit(x_train, y_train, epochs=10)
#model.evaluate(x_test, y_test)
