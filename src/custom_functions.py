from imports import *
def mean_sqr_error(y_true, y_pred): # it simplifies gradient descend
	return  (1.0*K.mean(K.square(y_pred - y_true), axis=None))
