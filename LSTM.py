#!/usr/bin/env python
# coding: utf-8

# # Neural Networks Architecture for Decoding EEG MI Data using Spectrogram Representations
# In case that gumpy is not installed as a module, we need to specify the path to ``gumpy``. In addition, we wish to configure jupyter notebooks and any backend properly. Note that it may take some time for ``gumpy`` to load due to the number of dependencies


from __future__ import print_function

import os

os.environ["THEANO_FLAGS"] = "device=gpu0"
import sys

sys.path.append('./gumpy')

import gumpy
import numpy as np
import utils
from hyperopt import hp, Trials, STATUS_OK
from sklearn.metrics import confusion_matrix, cohen_kappa_score
from keras.models import load_model

# ## Setup parameters for the model and data
# Before we jump into the processing, we first wish to specify some parameters (e.g. frequencies) that we know from the data.

DEBUG = True
CLASS_COUNT = 2

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

# ## Load raw data
#
# Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.

# In[4]:


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

# ## Augment data

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

# # LSTM

from keras.layers import Dense, LSTM as _LSTM


def LSTM_model(input_shape, num_hidden_neurons=128,
               num_layers=1, dropout=0.2, recurrent_dropout=0.2,
               print_summary=False):
    model = Sequential()
    if num_layers > 1:
        for i in range(1, num_layers, 1):
            model.add(_LSTM(num_hidden_neurons, input_shape=input_shape,
                            return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout))
        model.add(_LSTM(num_hidden_neurons))
    else:
        model.add(_LSTM(num_hidden_neurons, input_shape=input_shape, dropout=dropout,
                        recurrent_dropout=recurrent_dropout))
    model.add(Dense(2, activation='softmax'))

    if print_summary:
        print(model.summary())

        # compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # assign and return

    return model


from sklearn.model_selection import StratifiedKFold


def train_model(params):
    neurons_per_layer, n_hidden_layers, dropout = int(params["neurons_per_layer"]), int(params["n_hidden_layers"]), \
                                                  params["dropout"]

    print("Training LSTM with {} h_neurons, {} h_layers and {:.2f} recurrent dropout".format(
        neurons_per_layer, n_hidden_layers, dropout))

    # define KFOLD-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    cvscores = []
    test_set_acc = 0
    ii = 1
    conf_matrix_testing = None
    conf_matrix_training = None
    training_kappas = []
    testing_kappas = []
    for train, test in kfold.split(x_subject, y_subject[:, 0]):
        # create callbacks
        model_name_str = "ModelSave/" + 'GRAZ_LSTM_' + '_run_' + str(ii)

        # checkpoint = ModelCheckpoint(model_name_str, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # callbacks_list = [checkpoint]
        # Fit the model
        # initialize and create the model
        model = LSTM_model(x_subject.shape[1:], num_layers=n_hidden_layers, num_hidden_neurons=neurons_per_layer,
                           dropout=dropout, print_summary=False)

        # fit model. If you specify monitor=True, then the model will create callbacks
        # and write its state to a HDF5 file
        # NumbItr = int(neurons_per_layer * (n_hidden_layers + 1) / 30)
        NumbItr = 10

        model.fit(x_subject[train], y_subject[train],
                  epochs=NumbItr,
                  batch_size=256,
                  verbose=0,
                  validation_split=0.1)

        # evaluate the model
        val_acc = model.evaluate(x_subject[test], y_subject[test], verbose=0)[1]
        print("Validation accuracy on run {}/{}: {:.2f}".format(ii, KFOLD, val_acc * 100))
        cvscores.append(val_acc * 100)

        # useful for metrics
        true_labels_train = np.array(y_subject[train]).argmax(axis=-1)
        pred_labels_train = model.predict(x_subject[train]).argmax(axis=-1)
        true_labels_test = np.array(y_subject[test]).argmax(axis=-1)
        pred_labels_test = model.predict(x_subject[test]).argmax(axis=-1)

        # calc confusion matrices
        conf_matrix_testing = confusion_matrix(true_labels_test, pred_labels_test)
        conf_matrix_training = confusion_matrix(true_labels_train, pred_labels_train)

        # calc kappa coefficients
        testing_kappa = cohen_kappa_score(true_labels_test, pred_labels_test)
        training_kappa = cohen_kappa_score(true_labels_train, pred_labels_train)
        print("Training kappa: {:.3f}  Testing kappa: {:.3f}".format(training_kappa, testing_kappa))
        training_kappas.append(training_kappa)
        testing_kappas.append(testing_kappa)

        if ii == kfold:
            model.save('LSTM_no_spectrogram_.h5')  # save the model for future use
            print("Model saved to disk!")

        ii += 1

    # print some evaluation statistics and write results to file
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # cv_all_subjects = np.asarray(cvscores)
    # print('Saving CV values to file....')
    # np.savetxt("Results/" + 'GRAZ_CV_' + 'LSTM_' + str(DROPOUT) + 'do' + '.csv',
    #            cv_all_subjects, delimiter=',', fmt='%2.4f')
    # print('CV values successfully saved!\n')
    return {'loss': 100.0 - np.mean(cvscores), 'n_hidden_neurons': neurons_per_layer,
            'n_hidden_layers': n_hidden_layers, 'dropout': dropout, 'status': STATUS_OK,
            'avg_validation_acc': np.mean(cvscores), 'conf_matrix_training': conf_matrix_training,
            "conf_matrix_testing": conf_matrix_testing, "training_kappas": training_kappas,
            "testing_kappas": testing_kappas}


space = {'neurons_per_layer': hp.quniform('neurons_per_layer', 10, 500, 10),
         'n_hidden_layers': hp.choice('n_hidden_layers', [1, 2, 5]),
         'dropout': hp.uniform('dropout', 0.1, 1.0)
         }

bayes_trials = Trials()

MAX_EVALS = 8

# Optimize (comment back in to run optimization)
# best = fmin(fn=objective, space=space, algo=tpe.suggest,
#             max_evals=MAX_EVALS, trials=bayes_trials)
#
# print("The results are:\n", best)

# comment out if running optimization again
best = {'dropout': 0.49299324051304194, 'n_hidden_layers': 2, 'neurons_per_layer': 290.0}

res = train_model(best)
print("Average validation accuracy: {}".format(res['avg_validation_acc']))
print("Confusion matrix of last fold (training):  with kappa: {}".format(res["training_kappas"][-1]))
print(res["conf_matrix_training"])
print("Confusion matrix of last fold (testing):  with kappa: {}".format(res["testing_kappas"][-1]))
print(res["conf_matrix_testing"])

# model2 = load_model('LSTM_no_spectrogram_.h5')

