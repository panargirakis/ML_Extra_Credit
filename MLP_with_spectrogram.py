#!/usr/bin/env python
# coding: utf-8

# # Neural Networks Architecture for Decoding EEG MI Data using Spectrogram Representations

# ## Preparation
# In case that gumpy is not installed as a module, we need to specify the path to ``gumpy``. In addition, we wish to configure jupyter notebooks and any backend properly. Note that it may take some time for ``gumpy`` to load due to the number of dependencies

from __future__ import print_function
import os
os.environ["THEANO_FLAGS"] = "device=gpu0"
import sys
sys.path.append('./gumpy')
import gumpy
import numpy as np
import utils
from sklearn.metrics import confusion_matrix

utils.print_version_info()

# ## Setup parameters for the model and data
# Before we jump into the processing, we first wish to specify some parameters (e.g. frequencies) that we know from the data.

DEBUG = True
CLASS_COUNT = 2
DROPOUT = 0.2  # dropout rate in float

# parameters for filtering data
FS = 250
LOWCUT = 2
HIGHCUT = 60
ANTI_DRIFT = 0.5
CUTOFF = 50.0  # freq to be removed from signal (Hz) for notch filter
Q = 30.0  # quality factor for notch filter
W0 = CUTOFF / (FS / 2)
AXIS = 0

# set random seed
SEED = 42
KFOLD = 5
n_hidden_layers = 2
neurons_per_layer = 100
NumbItr = int(neurons_per_layer * n_hidden_layers / 20)

# ## Load raw data
# Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.

# specify the location of the GrazB datasets
data_dir = './data/Graz'
subject = 'B01'

# initialize the data-structure, but do _not_ load the data yet
grazb_data = gumpy.data.GrazB(data_dir, subject)

# now that the dataset is setup, we can load the data. This will be handled from within the utils function,
# which will first load the data and subsequently filter it using a notch and a bandpass filter.
# the utility function will then return the training data.
x_train, y_train = utils.load_preprocess_data(grazb_data, True, LOWCUT, HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF,
                                              AXIS, FS)

x_augmented, y_augmented = gumpy.signal.sliding_window(data=x_train[:, :, :],
                                                       labels=y_train[:, :],
                                                       window_sz=4 * FS,
                                                       n_hop=FS // 10,
                                                       n_start=FS * 1)
x_subject = x_augmented
y_subject = y_augmented
x_subject = np.rollaxis(x_subject, 2, 1)

# from .model import KerasModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import Dropout
from kapre.utils import Normalization2D
from kapre.time_frequency import Spectrogram

import kapre


def MLP_model(input_shape, dropout=0.5, print_summary=False):
    # basis of the CNN_STFT is a Sequential network
    model = Sequential()

    # spectrogram creation using STFT
    model.add(Spectrogram(n_dft = 128, n_hop = 16, input_shape = input_shape,
              return_decibel_spectrogram = False, power_spectrogram = 2.0,
              trainable_kernel = False, name = 'static_stft'))
    model.add(Normalization2D(str_axis = 'freq'))
    model.add(Flatten())
    model.add(Dense(neurons_per_layer, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))

    # custom number of hidden layers
    for each in range(n_hidden_layers - 1):
        model.add(Dense(neurons_per_layer, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(2))  # two classes only
    model.add(Activation('softmax'))

    if print_summary:
        print(model.summary())

    # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # assign model and return

    return model



from sklearn.model_selection import StratifiedKFold
from keras.callbacks import ModelCheckpoint

# define KFOLD-fold cross validation test harness
kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
cvscores = []
ii = 1
conf_matrix_training = None
conf_matrix_testing = None
for train, test in kfold.split(x_subject, y_subject[:, 0]):
    print('Run ' + str(ii) + '...')
    # create callbacks
    model_name_str = "ModelSave/" + 'GRAZ_MLP_' + '_run_' + str(ii)

    checkpoint = ModelCheckpoint(model_name_str, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    # Fit the model
    # initialize and create the model
    model = MLP_model(x_subject.shape[1:], dropout=DROPOUT, print_summary=False)

    # fit model. If you specify monitor=True, then the model will create callbacks
    # and write its state to a HDF5 file
    model.fit(x_subject[train], y_subject[train],
              epochs=NumbItr,
              batch_size=256,
              verbose=1,
              validation_split=0.1, callbacks=callbacks_list)

    # evaluate the model
    print('Evaluating model on test set...')
    scores = model.evaluate(x_subject[test], y_subject[test], verbose=0)
    print("Result on test set: %s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
    cvscores.append(scores[1] * 100)
    conf_matrix_testing = confusion_matrix(np.array(y_subject[test]).argmax(axis=-1),
                                           model.predict(x_subject[test]).argmax(axis=-1))
    conf_matrix_training = confusion_matrix(np.array(y_subject[train]).argmax(axis=-1),
                                            model.predict(x_subject[train]).argmax(axis=-1))
    ii += 1

# print some evaluation statistics and write results to file
print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
cv_all_subjects = np.asarray(cvscores)
# print('Saving CV values to file....')
# np.savetxt("Results/" + 'GRAZ_CV_' + 'MLP_' + str(DROPOUT) + 'do' + '.csv',
#            cv_all_subjects, delimiter=',', fmt='%2.4f')
# print('CV values successfully saved!\n')

print("Confusion matrix of last fold (training):")
print(conf_matrix_training)
print("Confusion matrix of last fold (testing):")
print(conf_matrix_testing)

# from keras.models import load_model
#
# model.save('ModelSave/' + 'MLPmonitoring.h5')  # creates a HDF5 file 'my_model.h5'
# model2 = load_model('ModelSave/' + 'MLPmonitoring.h5',
#                     custom_objects={'Spectrogram': kapre.time_frequency.Spectrogram,
#                                     'Normalization2D': kapre.utils.Normalization2D})
