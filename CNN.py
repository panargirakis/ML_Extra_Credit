from __future__ import print_function

import os

os.environ["THEANO_FLAGS"] = "device=gpu0"
import sys

sys.path.append('./gumpy')

import gumpy
import numpy as np
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe
from keras.models import load_model

#
# To use the models provided by `gumpy-deeplearning`, we have to set the path to the models directory and import it. If you installed `gumpy-deeplearning` as a module, this step may not be required.

# ## Utility functions
#
# The examples for ``gumpy-deeplearning`` ship with a few tiny helper functions. For instance, there's one that tells you the versions of the currently installed keras and kapre. ``keras`` is required in ``gumpy-deeplearning``, while ``kapre``
# can be used to compute spectrograms.
#
# In addition, the utility functions contain a method ``load_preprocess_data`` to load and preprocess data. Its usage will be shown further below
import utils

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
NumbItr = 10

# ## Load raw data
#
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

# ## Augment data

# In[5]:


x_augmented, y_augmented = gumpy.signal.sliding_window(data=x_train[:, :, :],
                                                       labels=y_train[:, :],
                                                       window_sz=4 * FS,
                                                       n_hop=FS // 10,
                                                       n_start=FS * 1)
x_subject = x_augmented
y_subject = y_augmented
x_subject = np.rollaxis(x_subject, 2, 1)

# # CNN model

# In[6]:


# from .model import KerasModel
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers import BatchNormalization, Dropout, Conv1D, MaxPooling1D
from sklearn.metrics import confusion_matrix, cohen_kappa_score


def CNN_model(input_shape, filter1_s=96, filter2_s=64, filter3_s=128, kernel_size=3, dropout=0.5, print_summary=False):
    # basis of the CNN_STFT is a Sequential network
    model = Sequential()

    # Conv Block 1
    model.add(Conv1D(filters=filter1_s, kernel_size=kernel_size, name='conv1.2',
                     activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling1D(pool_size=2, padding='valid',
                           data_format='channels_last'))

    # Conv Block 2
    model.add(Conv1D(filters=filter2_s, kernel_size=kernel_size,
                     name='conv2', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling1D(pool_size=2, padding='valid',
                           data_format='channels_last'))

    # Conv Block 3
    model.add(Conv1D(filters=filter3_s, kernel_size=kernel_size,
                     name='conv3', activation='relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling1D(pool_size=2,
                           padding='valid',
                           data_format='channels_last'))
    model.add(Dropout(dropout))

    # classifier
    model.add(Flatten())
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


def train_model(params):
    filter1_s, filter2_s, filter3_s, kernel_size = int(params['filter1_s']), int(params['filter2_s']), \
                                                   int(params['filter3_s']), int(params['kernel_size'])

    print(
        "Training CNN with {} filters in block 1, {} filters in block 2, {} filters in block 3 and kernel sizes of {}".format(
            filter1_s, filter2_s, filter3_s, kernel_size))

    # define KFOLD-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=SEED)
    cvscores = []
    conf_matrix_training = None
    conf_matrix_testing = None
    training_kappas = []
    testing_kappas = []
    ii = 1
    for train, test in kfold.split(x_subject, y_subject[:, 0]):
        # create callbacks
        model_name_str = "ModelSave/" + 'GRAZ_CNN_STFT_3layer_' + \
                         '_run_' + str(ii)

        # checkpoint = ModelCheckpoint(model_name_str, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        # callbacks_list = [checkpoint]
        # Fit the model
        # initialize and create the model
        model = CNN_model(x_subject.shape[1:], filter1_s, filter2_s, filter3_s, kernel_size, dropout=DROPOUT,
                          print_summary=False)

        # fit model. If you specify monitor=True, then the model will create callbacks
        # and write its state to a HDF5 file
        num_samples, row_num, col_num = x_subject[train].shape
        x_train_reshaped = np.reshape(x_subject[train], (num_samples, col_num, row_num))
        model.fit(x_train_reshaped, y_subject[train],
                  epochs=NumbItr,
                  batch_size=256,
                  verbose=0,
                  validation_split=0.1)

        # evaluate the model
        num_samples, row_num, col_num = x_subject[test].shape
        test_reshaped = np.reshape(x_subject[test], (num_samples, col_num, row_num))
        val_acc = model.evaluate(test_reshaped, y_subject[test], verbose=0)[1]
        print("Validation accuracy on run {}/{}: {:.2f}".format(ii, KFOLD, val_acc * 100))
        cvscores.append(val_acc * 100)

        # useful for metrics
        true_labels_train = np.array(y_subject[train]).argmax(axis=-1)
        pred_labels_train = model.predict(x_train_reshaped).argmax(axis=-1)
        true_labels_test = np.array(y_subject[test]).argmax(axis=-1)
        pred_labels_test = model.predict(test_reshaped).argmax(axis=-1)

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
            model.save('CNN_no_spectrogram_.h5')  # save the model for future use
            print("Model saved to disk!")

        ii += 1

    # print some evaluation statistics and write results to file
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # cv_all_subjects = np.asarray(cvscores)
    # print('Saving CV values to file....')
    # np.savetxt("Results/" + 'GRAZ_CV_' + 'CNN_STFT_3layer_' + str(DROPOUT) + 'do' + '.csv',
    #            cv_all_subjects, delimiter=',', fmt='%2.4f')
    # print('CV values successfully saved!\n')
    return {'loss': 100.0 - np.mean(cvscores), 'filter1_s': filter1_s,
            'filter2_s': filter2_s, 'filter3_s': filter3_s, 'kernel_size': kernel_size, 'status': STATUS_OK,
            'avg_validation_acc': np.mean(cvscores), 'conf_matrix_training': conf_matrix_training,
            "conf_matrix_testing": conf_matrix_testing, "training_kappas": training_kappas,
            "testing_kappas": testing_kappas}


space = {'filter1_s': hp.quniform('filter1_s', 30, 150, 10),
         'filter2_s': hp.quniform('filter2_s', 30, 150, 10),
         'filter3_s': hp.quniform('filter3_s', 30, 150, 10),
         'kernel_size': hp.quniform('kernel_size', 3, 6, 1)
         }

bayes_trials = Trials()

MAX_EVALS = 8

# Optimize (comment back in to run optimization)
# best = fmin(fn=train_model, space=space, algo=tpe.suggest,
#             max_evals=MAX_EVALS, trials=bayes_trials)
#
# print("The results are:\n", best)

# comment out if running optimization
best = {'filter1_s': 90.0, 'filter2_s': 40.0, 'filter3_s': 60.0, 'kernel_size': 6.0}

res = train_model(best)
print("Average validation accuracy: {}".format(res['avg_validation_acc']))
print("Confusion matrix of last fold (training):  with kappa: {}".format(res["training_kappas"][-1]))
print(res["conf_matrix_training"])
print("Confusion matrix of last fold (testing):  with kappa: {}".format(res["testing_kappas"][-1]))
print(res["conf_matrix_testing"])

# model2 = load_model('CNN_no_spectrogram_.h5')
