#!/usr/bin/env python

BITRATE = 16                      # Bit rate
SAMRATE = 16000                   # Sample rate (Hz)
SAMTIME = 1000.0 / SAMRATE        # Sample time (ms)
MAXAMPS = float(2**BITRATE / 2)   # Max samples amplitute

DATA_DIR = './data'               # Data location
OUT_DIR = './models'              # Model output location

LABELS_REQUIRED = ['yes', 'no', 'up', 'down', 'left',
                   'right', 'on', 'off', 'stop', 'go',
                   'silence']

NUM2LABEL = {i + 1: l for i, l in enumerate(LABELS_REQUIRED)}   # ID to label dictionary
NUM2LABEL[0] = 'unknown'
LABEL2NUM = {v: k for k, v in NUM2LABEL.items()}                # Label to ID dictionary
N_CLASSES = len(LABEL2NUM)                                      # Number of classes/labels
