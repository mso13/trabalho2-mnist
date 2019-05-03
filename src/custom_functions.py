from imports import *

from keras import losses

def mean_sqr_error(y_true, y_pred): # it simplifies gradient descend
	print('True')
	print(y_true)
	print('Pred')
	print(y_pred)
	print('cost')
	print(1.0*K.mean(K.square(y_pred - y_true), axis=None))
	print('test')
	print(K.mean(K.square(y_pred - y_true), axis=None))
	#return  (1.0*K.mean(K.square(y_pred - y_true), axis=None))
	return  (losses.mean_squared_error(y_true, y_pred))
