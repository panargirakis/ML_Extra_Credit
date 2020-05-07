from __future__ import print_function
import sys

sys.path.append('./gumpy')

import gumpy
import numpy as np
import utils
from hyperopt import hp, tpe, Trials, fmin, STATUS_OK
import plotly.express as px
import pandas as pd
from scipy import signal
from scipy.fftpack import fftshift
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import plotly.offline as pt


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
# Before training and testing a model, we need some data. The following code shows how to load a dataset using ``gumpy``.


# specify the location of the GrazB datasets
data_dir = './data/Graz'
subject = 'B01'

# initialize the data-structure, but do _not_ load the data yet
grazb_data = gumpy.data.GrazB(data_dir, subject)

# now that the dataset is setup, we can load the data. This will be handled from within the utils function,
# which will first load the data and subsequently filter it using a notch and a bandpass filter.
# the utility function will then return the training data.
x_train, y_train = utils.load_preprocess_data(grazb_data, True, LOWCUT,
                                              HIGHCUT, W0, Q, ANTI_DRIFT, CLASS_COUNT, CUTOFF,
                                              AXIS, FS)

# ## Augment data

# x_augmented, y_augmented = gumpy.signal.sliding_window(data=x_train[:, :, :],
#                                                        labels=y_train[:, :],
#                                                        window_sz=4 * FS,
#                                                        n_hop=FS // 10,
#                                                        n_start=FS * 1)
vis_x = np.rollaxis(x_train, 2, 1)  # make sure it is "run", "sensor" "sample"
df = pd.DataFrame({"signal_avg_0": vis_x[0][0] + vis_x[1][0] + vis_x[2][0] + vis_x[3][0] + vis_x[4][0]
                             + vis_x[5][0] + vis_x[6][0] + vis_x[7][0] + vis_x[8][0] + vis_x[9][0]
                   })
df["signal_avg_0"] /= 10
df["signal_avg_1"] = vis_x[-1][0] + vis_x[-2][0] + vis_x[-3][0] + vis_x[-4][0] + vis_x[-5][0] + vis_x[-6][0] + vis_x[-7][0] + vis_x[-8][0] + vis_x[-9][0] + vis_x[-10][0]
df["signal_avg_1"] /= 10
df["signal_0"] = vis_x[0][0]
df["signal_1"] = vis_x[-1][0]


plt.figure(figsize=[5, 5])
plt.title("Average signal of 10 recordings with y=0")
# default is power spectral density
plt.specgram(df["signal_avg_0"], NFFT=512, Fs=FS, scale="linear", noverlap=64, detrend="linear")
plt.ylim(top=100)
plt.yticks(np.arange(0, 100, step=5))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig("Results/spectrogram_mpl_avg_y0.png")

plt.clf()

plt.figure(figsize=[5, 5])
plt.title("Signal a recording with y=0")
# default is power spectral density
plt.specgram(df["signal_0"], NFFT=512, Fs=FS, scale="linear", noverlap=64, detrend="linear")
plt.ylim(top=100)
plt.yticks(np.arange(0, 100, step=5))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig("Results/spectrogram_mpl_y0.png")

plt.clf()

plt.figure(figsize=[5, 5])
plt.title("Average signal of 10 recordings with y=1")
# default is power spectral density
plt.specgram(df["signal_avg_1"], NFFT=512, Fs=FS, scale="linear", noverlap=64, detrend="linear")
plt.ylim(top=100)
plt.yticks(np.arange(0, 100, step=5))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig("Results/spectrogram_mpl_avg_y1.png")

plt.clf()

plt.figure(figsize=[5, 5])
plt.title("Signal a recording with y=1")
# default is power spectral density
plt.specgram(df["signal_1"], NFFT=512, Fs=FS, scale="linear", noverlap=64, detrend="linear")
plt.ylim(top=100)
plt.yticks(np.arange(0, 100, step=5))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig("Results/spectrogram_mpl_y1.png")
