from imports import *

#from keras import losses

def sqr_error(y_true, y_pred): # it simplifies gradient descend
	return  K.sum(K.square(y_pred - y_true), axis=None)
