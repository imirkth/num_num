"""
Utility used by the Network class to actually train.

Based on:
    https://github.com/fchollet/keras/blob/master/examples/mnist_mlp.py

"""
from __future__ import print_function

from sklearn.model_selection import GroupKFold
from keras.datasets import mnist, cifar10
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
import pandas as pd
from keras.layers.core import Dense, Dropout
from keras.models import Sequential, load_model
from keras.layers.normalization import BatchNormalization
from sklearn.decomposition import PCA
import consistency as cs

import gc
# Helper: Early stopping.
early_stopper = EarlyStopping(patience=5)

def get_cifar10():
    """Retrieve the CIFAR dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 64
    input_shape = (3072,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train = x_train.reshape(50000, 3072)
    x_test = x_test.reshape(10000, 3072)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)

def get_mnist():
    """Retrieve the MNIST dataset and process the data."""
    # Set defaults.
    nb_classes = 10
    batch_size = 128
    input_shape = (784,)

    # Get the data.
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return (nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test)


def load_data():
    print("Loading data...")
    # Load the data from the CSV files
    training_data = pd.read_csv('/home/imirk/PycharmProjects/AI_trading/rounds/complete_training_data.csv', header=0)
    prediction_data = pd.read_csv('/home/imirk/PycharmProjects/AI_trading/rounds/validation_data.csv', header=0)

    validation_data = prediction_data[prediction_data.data_type == 'validation']
    complete_training_data = pd.concat([training_data])#, validation_data])

    # Transform the loaded CSV data into numpy arrays
    features = [f for f in list(training_data) if "feature" in f]
    targets = [f for f in list(training_data) if "target" in f]
    x_train = complete_training_data[features]
    y_train = complete_training_data['target_bernie']
    y_train = pd.DataFrame(y_train)
    y_train.reset_index(inplace=True, drop=True)

    x_test = prediction_data[features]
    y_test = prediction_data['target_bernie'].fillna(-1)
    # y_test = y_test['target_bernie']

    ids = prediction_data["id"]
    extra = 'auto'  # auto
    if extra == 'pca':
        encoder = PCA(n_components=5)
        zenc = encoder.fit_transform(x_train)
        zenc2 = encoder.transform(x_test)
    else:
        encoder = load_model('/home/imirk/PycharmProjects/AI_trading/rounds/auto_enc.h5')
        zenc = encoder.predict(x_train.values)
        zenc2 = encoder.predict(x_test.values)
        # encoder = ld.encoder(x_train, x_test)
        #
    zenc_df = pd.DataFrame(zenc)
    x_train.reset_index(inplace=True, drop=True)
    zenc_df.reset_index(inplace=True, drop=True)
    x_train = pd.concat([x_train, zenc_df], axis=1)

    # zenc2 = encoder.predict(x_test)
    zenc2_df = pd.DataFrame(zenc2)
    x_test.reset_index(inplace=True, drop=True)
    zenc2_df.reset_index(inplace=True, drop=True)
    x_test = pd.concat([x_test, zenc2_df], axis=1)

    eras = complete_training_data['era']
    eras_validates = validation_data['era']

    y_test = pd.concat([y_test, eras_validates], axis=1)
    x_test = pd.concat([x_test, eras_validates], axis=1)

    nb_classes = 1
    batch_size = 1024
    input_shape = 55
    print(x_train.shape)
    return nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, ids, eras


def compile_model(network, nb_classes, input_shape):
    """Compile a sequential model.

    Args:
        network (dict): the parameters of the network

    Returns:
        a compiled network.

    """
    # Get our network parameters.
    nb_layers = network['nb_layers']
    nb_neurons = [network['nb_neurons_l0'], network['nb_neurons_l1'], network['nb_neurons_l2'], network['nb_neurons_l3']]
    d_rate = [network['dr0'], network['dr1'], network['dr2'], network['dr3']]
    activation = network['activation']
    optimizer = network['optimizer']

    model = Sequential()

    # Add each layer.
    for i in range(nb_layers):

        # Need input shape for first layer.
        if i == 0:
            if type(input_shape) == type(0):
                model.add(Dense(nb_neurons[i], activation=activation, input_dim=input_shape))
                model.add(BatchNormalization())
                model.add(Dropout(d_rate[i]))
            else:
                model.add(Dense(nb_neurons[i], activation=activation, input_shape=input_shape))
                model.add(BatchNormalization())
                model.add(Dropout(d_rate[i]))
        else:
            model.add(Dense(nb_neurons[i], activation=activation))
            model.add(BatchNormalization())
            model.add(Dropout(d_rate[i]))

    # Output layer.
    model.add(Dense(nb_classes, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=optimizer,
                  metrics=['accuracy'])

    return model

def train_and_score(network, nb_classes, batch_size, input_shape, x_train, x_test, y_train, y_test, eras):
    """Train the model, return test loss.

    Args:
        network (dict): the parameters of the network
        nb_classes,
        batch_size,
        input_shape,
        x_train,
        x_test,
        y_train,
        y_test,
        eras
        those are Dataset to use for training/evaluating

    """

    model = compile_model(network, nb_classes, input_shape)

    gkf = GroupKFold(n_splits=10)
    kfold_split = gkf.split(x_train, y_train, groups=eras)
    #we split the training in folds and we train and test on the folds
    for index_train, index_test in kfold_split:
        print(len(index_train), len(index_test))
        X_train, X_test = x_train.loc[index_train], x_train.loc[index_test]
        Y_train, Y_test = y_train.loc[index_train], y_train.loc[index_test]
        print(X_train.shape, Y_train.shape)
        model.fit(X_train.values, Y_train.values, batch_size=batch_size, epochs=10,
                     validation_data=(X_test.values, Y_test.values), verbose=2, callbacks=[early_stopper])


    score = model.evaluate(x_test[:46362].drop('era', axis=1).values, y_test[:46362].drop('era', axis=1).values, verbose=0)
    consistency = cs.check_consistency(model, x_test, y_test, x_test['era'][:46362])
    model = None
    gc.collect()
    return (score[0], 1- consistency) # 1 is accuracy. 0 is loss.
    # i want to have decreasing loss and consistency equal to zero as well
