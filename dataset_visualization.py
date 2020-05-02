
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

x_augmented, y_augmented = gumpy.signal.sliding_window(data=x_train[:, :, :],
                                                       labels=y_train[:, :],
                                                       window_sz=4 * FS,
                                                       n_hop=FS // 10,
                                                       n_start=FS * 1)
vis_x = np.rollaxis(x_train, 2, 1)  # make sure it is "run", "sensor" "sample"
df = pd.DataFrame({"signal": vis_x[0][0]})
df["second"] = df['signal'].index / (2000/8)
fig = px.line(df, x="second", y="signal", title="Raw signal of 1 recording")
# fig.show()

# issue: to go down to 1Hz, a window of 1250 is needed
# window size = 5 * sampling rage / lowest freq (156 for down to 8 hz, start of alpha waves)
freqs, bins, Pxx = signal.spectrogram(df["signal"], FS, nperseg=125)
plt.pcolormesh(bins, freqs, Pxx)  # show in log scale (do average first then log)
plt.ylim(top=100)
plt.yticks(np.arange(0, 100, step=5))
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.savefig("Results/spectrogram.png")


print("Filtered data shape: {}".format(x_train.shape))
print("Augmented data shape: {}".format(x_augmented.shape))
