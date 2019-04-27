import tensorflow as tf
from keras import backend as K # para implementar funcoes de ativacao personalizadas
import os
# referencia para adicionar funcoes de ativacao personalizadas: 
#https://stackoverflow.com/questions/43915482/how-do-you-create-a-custom-activation-function-with-keras
#tentar conda install tensorflow-gpu caso esteja usando anaconda e ocorra erro de dll
tf.enable_eager_execution()
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def custom_activation(x):
#sigmoide quadrada para reduzir a ativacao do neuronio caso as entradas sejam inibitorias (negativas ou proximas a 0)
    return K.sigmoid(x)*K.sigmoid(x)

mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
#print(type(x_train))
#print(type(x_test))
#print(x_train[0,:,:])
#print(x_train[0,:,:].shape)
#print(x_test[0,:,:])
#print(x_test[0,:,:].shape)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(60, activation=custom_activation),
  tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)
model.evaluate(x_test, y_test)
