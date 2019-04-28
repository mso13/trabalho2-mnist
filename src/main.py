# Código Fonte do Trabalho 2 - MNIST Dataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import mlp

def read_idx(filename):
	""" A function that can read MNIST's idx file format into numpy arrays.
    The MNIST data files can be downloaded from here:

    http://yann.lecun.com/exdb/mnist/
    This relies on the fact that the MNIST dataset consistently uses
    unsigned char types with their data segments.
	"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)


def main():
	# Importando as imagens para treinamento
	raw_train = read_idx('dados/train-images.idx3-ubyte')
	X_train   = np.reshape(raw_train, (60000, 28*28))
	y_train   = read_idx('dados/train-labels.idx1-ubyte')

	# Importanto as imagens para teste
	raw_test = read_idx('dados/t10k-images.idx3-ubyte')
	X_test   = np.reshape(raw_test, (10000, 28*28))
	y_test   = read_idx('dados/t10k-labels.idx1-ubyte')


if __name__ == '__main__':
	main()