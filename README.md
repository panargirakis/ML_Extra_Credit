# Introduction

This repository contains examples of different models to solve BCI Competition Graz 2b classification challenge.

The original work was done by Mohammad Reza Razei and was extended by Panos Argyrakis.

# Usage

Install all the requirements in requirements.txt:

```pip install -r requirements.txt```

Run any of:
- LSTM.py
- MLP.py
- CNN.py
- CNN_with_spectrogram.py
- MLP_with_spectrogmam.py

# Notes

The Hybrid.py network was an idea to combine a CNN with an LSTM to capture both spatial and temporal features. As of now, it does not execute.

The scripts automatically save the model from the last fold.

Some models (LSTM, MLP and CNN) are set up for automatic hyperparameter tuning. To enable, uncomment the correct lines at the bottom of the corresponding script.

The "Results" folder contains training logs, hyperparameter tuning logs and some visualizations of the dataset.
