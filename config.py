#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from dataload import loadtxt_gcp

# Aiudio file information
BITRATE = 16                      # Bit rate
SAMRATE = 16000                   # Sample rate (Hz)
SAMTIME = 1000.0 / SAMRATE        # Sample time (ms)
MAXAMPS = float(2**BITRATE / 2)   # Max samples amplitude

# Data location and model output location
DATA_DIR = './data'
OUT_DIR = './models'

# Noise file location
NOISE_DIR = os.path.join(DATA_DIR, 'train/audio/_background_noise_')
NOISE_CLIP_DIR = os.path.join(DATA_DIR, 'noise_clips')
NOISE_MATRIX = loadtxt_gcp(os.path.join(NOISE_CLIP_DIR,
                                        'noise_clips.csv'),
                           out='arr')

LABELS_REQUIRED = ['yes', 'no', 'up', 'down', 'left',
                   'right', 'on', 'off', 'stop', 'go',
                   'silence']

# Label number and number to label dictionaries
NUM2LABEL = {i + 1: l for i, l in enumerate(LABELS_REQUIRED)}
NUM2LABEL[0] = 'unknown'
LABEL2NUM = {v: k for k, v in NUM2LABEL.items()}
N_CLASSES = len(LABEL2NUM)
