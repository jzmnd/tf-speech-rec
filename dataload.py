#!/usr/bin/env python
"""
Functions for creating audio file dataframe and loading files and batches as tensors.
"""

import os
import re
import config as cfg
import pandas as pd

FILENAME_PATTERN = re.compile(r"([^_]+)_nohash_([^\.]+)\.wav")


def load_data(data_dir):
    """Loads data into training and validation sets"""
    with open(os.path.join(data_dir, 'train', 'validation_list.txt'), 'r') as fin:
        validation_files = set([f.strip() for f in fin.readlines()])

    with open(os.path.join(data_dir, 'train', 'testing_list.txt'), 'r') as fin:
        testing_files = set([f.strip() for f in fin.readlines()])

    audio_dir = os.path.join(data_dir, 'train', 'audio')
    labels_req = set(cfg.LABELS_REQUIRED)

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
