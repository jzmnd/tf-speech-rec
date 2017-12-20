#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Functions for creating audio file dataframe and loading files and batches as tensors.
"""

import os
import re
import config as cfg
import pandas as pd
import numpy as np
import yaml
from scipy.io import wavfile

FILENAME_PATTERN = re.compile(r"([^_]+)_nohash_([^\.]+)\.wav")


def load_data(data_dir):
    """Loads data into training and validation sets.
       Input: data_dir: location of data_dir
       Return: dataframe of all files and their information"""

    # Open list of validation and test files and convert to set
    with open(os.path.join(data_dir, 'train', 'validation_list.txt'), 'r') as fin:
        validation_files = set([f.strip() for f in fin.readlines()])

    with open(os.path.join(data_dir, 'train', 'testing_list.txt'), 'r') as fin:
        testing_files = set([f.strip() for f in fin.readlines()])

    # Location of audio files
    audio_dir = os.path.join(data_dir, 'train', 'audio')

    # Set of labels
    labels_req = set(cfg.LABELS_REQUIRED)

    # Loop through all files in all directories and create dataframe entry
    file_list = []
    for d in os.listdir(audio_dir):
        files = os.listdir(os.path.join(audio_dir, d))
        reqlabel_flag = d in labels_req
        if reqlabel_flag:
            reqlabel = d
        else:
            reqlabel = 'unknown'

        for f in files:
            ffull = os.path.join(d, f)
            if ffull in validation_files:
                setlabel = 'val'
            elif ffull in testing_files:
                setlabel = 'test'
            else:
                setlabel = 'train'
            r = re.match(FILENAME_PATTERN, f)
            if r:
                file_list.append([ffull, d, r.group(1), int(r.group(2)),
                                  setlabel, reqlabel, reqlabel_flag])

    return pd.DataFrame(file_list, columns=['filepath', 'label', 'uid', 'uversion',
                                            'setlabel', 'reqlabel', 'reqlabelflag'])


def load_config(filename, path="."):
    """Loads yaml file and returns model parameters as python dictionary"""
    with open(os.path.join(path, filename), 'r') as f:
        modeldict = yaml.load(f)
    return modeldict['model'], modeldict['params']


def load_batch(df, data_dir, batch_size=100, silence_size=5, label='train',
               random=True, seed=None, w=0.0568, samples=16000):
    """Loads a batch of audio data files and returns the array of wave data.
       Also returns the truth values.
       If random is True then it selects a random batch else it selects a
       continuous batch starting at seed."""

    # Select the required set (train, val, test)
    df_req = df[df.setlabel == label]

    # Weights to allow for larger numbers of unknowns than other labels
    weights = np.where(df_req.reqlabelflag, 1.0, w)

    # Select the required rows of the dataframe (randomly or in order)
    non_silence_size = batch_size - silence_size
    if random:
        np.random.seed(seed)
        df_req_batch = df_req.sample(n=non_silence_size, weights=weights)
    else:
        if not seed:
            seed = 0
        df_req_batch = df_req.iloc[seed:seed + non_silence_size]

    # Select files and y_true values
    X_list = df_req_batch.filepath
    y_true = df_req_batch.reqlabel.map(cfg.LABEL2NUM).values

    # Add silence labels to end
    y_true = np.hstack([y_true, silence_size * [cfg.LABEL2NUM['silence']]])

    # 1-hot encode y_true labels
    y_true_onehot = np.eye(len(cfg.LABEL2NUM))[y_true]

    # Empty array of size (batch_size x samples)
    X = np.zeros([batch_size, samples])

    # Load each wave file and add it to the array
    for i, f in enumerate(X_list):
        sr, wave = wavfile.read(os.path.join(data_dir, 'train', 'audio', f))
        # Reshape all files to be same length (i.e. samples)
        wave.resize(samples)
        X[i] += wave

    # Shuffle
    idx = np.random.permutation(batch_size)
    X = X[idx]
    y_true_onehot = y_true_onehot[idx]

    return X, y_true_onehot
