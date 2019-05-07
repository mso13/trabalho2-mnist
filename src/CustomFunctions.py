import tensorflow as tf
from keras import backend as K

def sqr_error(y_true, y_pred): # it simplifies gradient descend
	return  K.sum(K.square(y_pred - y_true), axis=None)

#def custom_activation(x):
#sigmoide quadrada para reduzir a ativacao do neuronio caso as entradas sejam inibitorias (negativas ou proximas a 0)
#    return K.sigmoid(x)*K.sigmoid(x)