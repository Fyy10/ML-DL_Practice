import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.datasets import mnist
from keras.optimizers import Adam, SGD
from keras.utils import np_utils

"""(x_train, y_train), (x_test, y_test) = mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train/225
x_test = x_test/255

num_pixel = 28*28
x_train = x_train.reshape(x_train.shape[0], num_pixel)
x_test = x_test.reshape(x_test.shape[0], num_pixel)"""

def load_data():
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	number = 10000
	x_train = x_train[0:number]
	y_train = y_train[0:number]
	x_train = x_train.reshape(number, 28*28)
	x_test = x_test.reshape(x_test.shape[0], 28*28)
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	#convert class vectors to binary class matrics
	y_train = np_utils.to_categorical(y_train, 10)
	y_test = np_utils.to_categorical(y_test, 10)

	x_train = x_train/255
	x_test = x_test/255
	return(x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

model = Sequential()

model.add(Dense(input_dim = 28*28, units = 100, activation = 'relu'))

model.add(Dense(units = 100, activation = 'relu'))

model.add(Dense(units = 100, activation = 'relu'))

model.add(Dense(units = 10, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(x_train, y_train, batch_size = 2000, epochs = 20)

score = model.evaluate(x_test, y_test)

print '\nFinal accuracy:', score[1]
