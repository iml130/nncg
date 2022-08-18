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
from keras.layers import Flatten, MaxPooling2D, Convolution2D, SeparableConvolution2D, Dropout, Dense
from keras.models import Sequential
from applications.daimler.loader import load_imdb

def create_conv_model(shape, ConvolutionType=Convolution2D):
    conv_model = Sequential()
    conv_model.add(ConvolutionType(8, (3, 3), input_shape=shape,
                                activation='relu', padding='same'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(ConvolutionType(24, (3, 3), padding='same', activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(ConvolutionType(48, (4, 3), padding='same', activation='relu'))
    conv_model.add(MaxPooling2D(pool_size=(2, 2)))
    conv_model.add(Dropout(0.4))
    conv_model.add(Flatten())
    conv_model.add(Dense(2, activation='softmax'))
    return conv_model

def train(imgdb_path, model_path, use_separable=False):
    imdb = load_imdb(imgdb_path)
    x = imdb['images']
    y = imdb['y']

    input_shape = (x.shape[1], x.shape[2], 1)
    epochs = 30 if use_separable else 10

    # Select the current model here
    if use_separable:
        model = create_conv_model(input_shape, SeparableConvolution2D)
    else:
        model = create_conv_model(input_shape, Convolution2D)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(x, y, batch_size=1000, epochs=epochs, verbose=1, validation_split=0.05)
    model.save(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the network given ')

    parser.add_argument('-b', '--database-path', dest='imgdb_path',
                        help='Path to the image database to use for training. '
                            'Default is img.db in current folder.')
    parser.add_argument('-m', '--model-path', dest='model_path',
                        help='Store the trained model using this path. Default is model.h5.')
    parser.add_argument('-s', '--separable', dest='use_separable', action='store_true',
                        help='Whether to use separable convolution instead of regular ones. Default is regular (Conv2D).')

    args = parser.parse_args()

    imgdb_path = "img.db"
    model_path = "model.h5"
    use_separable = False

    if args.imgdb_path is not None:
        imgdb_path = args.imgdb_path

    if args.model_path is not None:
        model_path = args.model_path
    
    if args.use_separable is not None:
        use_separable = args.use_separable
    
    train(imgdb_path, model_path, use_separable)

