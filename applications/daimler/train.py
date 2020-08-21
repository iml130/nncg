#!/usr/bin/env python3

# Use plaidml if available, tf otherwise
USE_PLAIDML = True
if USE_PLAIDML:
    from importlib import util
    spam_spec = util.find_spec("plaidml")
    if spam_spec is not None:
        import plaidml.keras
        plaidml.keras.install_backend()
    else:
        print("PlaidML not found, using Tensorflow")
        USE_PLAIDML = False
import argparse
from keras.layers import Flatten, MaxPooling2D, Convolution2D, Dropout, Dense
from keras.models import Sequential
from applications.daimler.loader import load_imdb

parser = argparse.ArgumentParser(description='Train the network given ')

parser.add_argument('-b', '--database-path', dest='imgdb_path',
                    help='Path to the image database to use for training. '
                         'Default is img.db in current folder.')
parser.add_argument('-m', '--model-path', dest='model_path',
                    help='Store the trained model using this path. Default is model.h5.')

args = parser.parse_args()

imgdb_path = "img.db"
model_path = "model.h5"

if args.imgdb_path is not None:
    imgdb_path = args.imgdb_path

if args.model_path is not None:
    model_path = args.model_path

imdb = load_imdb(imgdb_path)
x = imdb['images']
y = imdb['y']

usual_model = Sequential()
usual_model.add(Convolution2D(4, (3, 3), input_shape=(x.shape[1], x.shape[2], 1),
                              activation='relu', padding='same'))
usual_model.add(MaxPooling2D(pool_size=(2, 2)))
usual_model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
usual_model.add(MaxPooling2D(pool_size=(2, 2)))
usual_model.add(Convolution2D(32, (3, 3), padding='same', activation='relu',))
usual_model.add(MaxPooling2D(pool_size=(4, 2)))
usual_model.add(Dropout(0.4))
usual_model.add(Convolution2D(2, (2, 2), activation='softmax'))
usual_model.add(Flatten())

dense_model = Sequential()
dense_model.add(Convolution2D(4, (3, 3), input_shape=(x.shape[1], x.shape[2], 1),
                              activation='relu', padding='same'))
dense_model.add(MaxPooling2D(pool_size=(2, 2)))
dense_model.add(Convolution2D(16, (3, 3), padding='same', activation='relu'))
dense_model.add(MaxPooling2D(pool_size=(2, 2)))
dense_model.add(Convolution2D(32, (3, 3), padding='same', activation='relu'))
dense_model.add(MaxPooling2D(pool_size=(2, 2)))
dense_model.add(Dropout(0.4))
dense_model.add(Flatten())
dense_model.add(Dense(2, activation='softmax'))

# Select the current model here
model = dense_model

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(x, y, batch_size=1000, epochs=10, verbose=1, validation_split=0.05)
model.save(model_path)
