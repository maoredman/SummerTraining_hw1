import numpy as np
import os

os.environ["THEANO_FLAGS"] = "device=gpu0"
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from utils import get_output

# Setting
batch_size = 128
nb_classes = 10
max_epochs = 20
nb_hidden_width = 256
nb_hidden_layers = 3

# Load the MNIST data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28*28)
X_test = X_test.reshape(10000, 28*28)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# Normalize value of each pixel between 0 to 1
X_train /= 255
X_test /= 255

# Convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

# Build DNN model
model = Sequential()
model.add(Dense(nb_hidden_width, activation='relu', input_shape=(28*28,)))
for n in xrange(nb_hidden_layers-1):
    model.add(Dense(nb_hidden_width, activation='relu'))
model.add(Dense(10, activation='softmax'))

model.summary()
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])

# Training
history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=max_epochs, verbose=1, validation_data=(X_test, Y_test))

# Get testing acc
score = model.evaluate(X_test, Y_test, verbose=0)
print 'Test accuracy:', score[1]

print 'Plot images and store them in directory "img" ...'
if not os.path.exists('./img'):
    os.makedirs('./img')
# Get result from each layer
for i in xrange(len(model.layers) - 1):
    get_output(i, model, 'hidden_layer_%d' % (i+1), X_test, Y_test)
get_output(len(model.layers)-1, model, 'output_layer', X_test, Y_test)
