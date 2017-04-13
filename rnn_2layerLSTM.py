import numpy as np

import json
from keras.layers.wrappers import *
from keras.layers.core import Dense,Dropout, Activation, Masking
from keras.layers.recurrent import LSTM
from keras.layers.convolutional import Convolution1D
from keras.optimizers import *
from keras.models import Sequential
import time
from keras.models import model_from_json


def save_lstm(model, save_weight, save_topo):
	json_string = model.to_json()
	model.save_weights(save_weight, overwrite = True)
	with open(save_topo, 'w') as outfile:
		json.dump(json_string, outfile)


def model_lstm(len_seq = 30, im_size = (40, 40), fc_size = 128,
		save_weight = 'untrained_weight.h5' , save_topo = 'untrained_topo.json',
		save_result = True, lr = 0.001, momentum = 0.6, decay = 0.0005,
		nesterov = True, rho = 0.9, epsilon = 1e-6, opt = 'sgd',
		load_cache = False, cnn = False, dict_size = 53, filter_len = 5):
	
	try:
		if load_cache:
			return read_lstm(weights_filename = save_weight,
					topo_filename = save_topo)
	except:
		pass

	start_time = time.time()

	#Starting LSTM Model here


	model = Sequential()


	
	model.add(Dense(fc_size, input_shape= (len_seq, im_size[0]*im_size[1])))

	
	# Masking layer

	model.add(Masking(mask_value = 0.0))

	#First LSTM layer

	model.add(LSTM(fc_size, return_sequences = True))

	# Second LSTM layer

	model.add(LSTM(fc_size, return_sequences = False))

	# Final Dense layer

	model.add(Dense(dict_size))

	#softmax layer
	model.add(Activation('softmax'))

	#Build and pass optimizer

	if opt == 'sgd':
		optimizer = SGD( lr = lr, momentum = momentum, decay = decay, nesterov = nesterov)

	model.compile(loss = 'categorical_crossentropy', optimizer = optimizer)

	end_time = time.time()


	print (" Total time for compilation %d" % (end_time - start_time))

	if save_result:
		save_lstm(model, save_weight, save_topo)

	return model


def train_lstm(model=None,
		X_train = [], y_train = [],
		X_test = [], y_test = [], batch_size = 100,
		iter_times = 7, show_accuracy = True,
		save_weight = 'trained_weight.h5',
		save_topo = 'trained_topo.json',
		save_result = True, validation_split = 0.1):

	if (not model) or len(X_train) == 0:
		print("Invalid input params")
		return

	start_time = time.time()

	print("Training the model")

	model.fit(X_train, y_train, batch_size = batch_size, nb_epoch = iter_times,
		validation_split=validation_split, show_accuracy = show_accuracy)

	end_time = time.time()

	score, acc = model.evaluate(X_test, y_test, batch_size = batch_size, 
					show_accuracy = show_accuracy)

	print("Test score", score)
	print("Test accuracy", acc)


	if save_result:
		save_lstm(model, save_weight, save_topo)

	return score, acc


def read_lstm(weights_filename = 'trained_weight.h5',
		topo_filename = 'trained_topo.json'):

	with open(topo_filename) as data_file:
		topo = json.load(data_file)
		model = mode_from_json(topo)
		model.load_weights(weights_filename)
		return model


def test():
	print(model_lstm(cnn = False, save_result = False))


test()
