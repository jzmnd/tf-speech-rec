#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Convolutional neural net models for processing audio features.
"""

import tensorflow as tf
import config as cfg


# Functions to initialize weights and biases
def weight_variable(shape, name):
    """Creates a variable of size shape with random small positive numbers"""
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    """Creates a variable of size shape with a constant small positive number"""
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial, name=name)


# Conv2d, max pooling, and dropout wrapper functions for simplicity (No padding)
def conv2d(x, W, sx=1, sy=1):
    return tf.nn.conv2d(x, W, strides=[1, sx, sy, 1], padding='VALID')


def max_pool_2d(x, k=2):
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='VALID')


def dropout(x, d, is_training):
    return tf.cond(is_training, lambda: tf.nn.dropout(x, d), lambda: x)


def tf_diff_axis(arr):
    """Equivalent of np.diff on final axis"""
    return arr[..., 1:] - arr[..., :-1]


# Function to select the required model
def buildModel(modelname):
    if modelname == 'convSpeechModelA':
        return convSpeechModelA
    elif modelname == 'convSpeechModelB':
        return convSpeechModelB
    elif modelname == 'convSpeechModelC':
        return convSpeechModelC
    elif modelname == 'convSpeechModelD':
        return convSpeechModelD
    else:
        return convSpeechModelA


# Models
def convSpeechModelA(x_mel_in, x_mfcc_in, x_zcr_in, x_rmse_in, dropout_prob=None, is_training=False):
    """Low latency conv model using only the Mel Spectrogram"""

    # ======================================================
    # Setup the parameters for the model
    # ======================================================

    # Mel Spectrogram input size
    t_size = 122
    f_size = 64

    # Parameters for Conv layer 1 filter
    filter_size_t = t_size
    filter_size_f = 8
    filter_count = 256
    filter_stride_t = 1
    filter_stride_f = 4

    # Paramaters for FC layers
    fc_output_channels_1 = 128
    fc_output_channels_2 = 128
    fc_output_channels_3 = cfg.N_CLASSES

    # Number of elements in the first FC layer
    fc_element_count = int(filter_count *
                           int(1 + (t_size - filter_size_t) / filter_stride_t) *
                           int(1 + (f_size - filter_size_f) / filter_stride_f))

    # ======================================================
    # Setup dictionaries containing weights and biases
    # ======================================================

    weights = {
        'wconv1': weight_variable([filter_size_t, filter_size_f, 1, filter_count], 'wconv1'),
        'wfc1': weight_variable([fc_element_count, fc_output_channels_1], 'wfc1'),
        'wfc2': weight_variable([fc_output_channels_1, fc_output_channels_2], 'wfc2'),
        'wfc3': weight_variable([fc_output_channels_2, fc_output_channels_3], 'wfc3'),
    }
    biases = {
        'bconv1': bias_variable([filter_count], 'bconv1'),
        'bfc1': bias_variable([fc_output_channels_1], 'bfc1'),
        'bfc2': bias_variable([fc_output_channels_2], 'bfc2'),
        'bfc3': bias_variable([fc_output_channels_3], 'bfc3'),
    }

    # ======================================================
    # Model definition and calculations
    # ======================================================

    # Normalize (L2) along the time axis
    x_mel_in_norm = tf.nn.l2_normalize(x_mel_in, 1, epsilon=1e-8)

    # Reshape input to [audio file number, time size, freq size, channel]
    x_mel_rs = tf.reshape(x_mel_in_norm, [-1, t_size, f_size, 1])

    # Layer 1: first Conv layer, BiasAdd and ReLU
    x_mel_1 = tf.nn.relu(conv2d(x_mel_rs, weights['wconv1'],
                                sx=filter_stride_t,
                                sy=filter_stride_f) + biases['bconv1'])

    # Dropout 1:
    x_mel_dropout_1 = dropout(x_mel_1, dropout_prob, is_training)

    # Flatten layers
    x_mel_1_rs = tf.reshape(x_mel_dropout_1, [-1, fc_element_count])

    # Layer 2: first FC layer
    x_mel_2 = tf.matmul(x_mel_1_rs, weights['wfc1']) + biases['bfc1']

    # Dropout 2:
    x_mel_dropout_2 = dropout(x_mel_2, dropout_prob, is_training)

    # Layer 3: second FC layer
    x_mel_3 = tf.matmul(x_mel_dropout_2, weights['wfc2']) + biases['bfc2']

    # Dropout 3:
    x_mel_dropout_3 = dropout(x_mel_3, dropout_prob, is_training)

    # Layer 4: third FC layer
    x_mel_output = tf.matmul(x_mel_dropout_3, weights['wfc3']) + biases['bfc3']

    return x_mel_output


def convSpeechModelB(x_mel_in, x_mfcc_in, x_zcr_in, x_rmse_in, dropout_prob=None, is_training=False):
    """Low latency conv model using only the MFCCs"""

    # ======================================================
    # Setup the parameters for the model
    # ======================================================

    # MFCC input size
    t_size = 122
    f_size = 20

    # Parameters for Conv layer 1 filter
    filter_size_t = t_size
    filter_size_f = 4
    filter_count = 256
    filter_stride_t = 1
    filter_stride_f = 4

    # Paramaters for FC layers
    fc_output_channels_1 = 128
    fc_output_channels_2 = 128
    fc_output_channels_3 = cfg.N_CLASSES

    # Number of elements in the first FC layer
    fc_element_count = int(filter_count *
                           int(1 + (t_size - filter_size_t) / filter_stride_t) *
                           int(1 + (f_size - filter_size_f) / filter_stride_f))

    # ======================================================
    # Setup dictionaries containing weights and biases
    # ======================================================

    weights = {
        'wconv1': weight_variable([filter_size_t, filter_size_f, 1, filter_count], 'wconv1'),
        'wfc1': weight_variable([fc_element_count, fc_output_channels_1], 'wfc1'),
        'wfc2': weight_variable([fc_output_channels_1, fc_output_channels_2], 'wfc2'),
        'wfc3': weight_variable([fc_output_channels_2, fc_output_channels_3], 'wfc3'),
    }
    biases = {
        'bconv1': bias_variable([filter_count], 'bconv1'),
        'bfc1': bias_variable([fc_output_channels_1], 'bfc1'),
        'bfc2': bias_variable([fc_output_channels_2], 'bfc2'),
        'bfc3': bias_variable([fc_output_channels_3], 'bfc3'),
    }

    # ======================================================
    # Model definition and calculations
    # ======================================================

    # Normalize (L2) along the time axis
    x_mfcc_in_norm = tf.nn.l2_normalize(x_mfcc_in, 1, epsilon=1e-8)

    # Reshape input to [audio file number, time size, freq size, channel]
    x_mfcc_rs = tf.reshape(x_mfcc_in_norm, [-1, t_size, f_size, 1])

    # Layer 1: first Conv layer, BiasAdd and ReLU
    x_mfcc_1 = tf.nn.relu(conv2d(x_mfcc_rs, weights['wconv1'],
                                 sx=filter_stride_t,
                                 sy=filter_stride_f) + biases['bconv1'])

    # Dropout 1:
    x_mfcc_dropout_1 = dropout(x_mfcc_1, dropout_prob, is_training)

    # Flatten layers
    x_mfcc_1_rs = tf.reshape(x_mfcc_dropout_1, [-1, fc_element_count])

    # Layer 2: first FC layer
    x_mfcc_2 = tf.matmul(x_mfcc_1_rs, weights['wfc1']) + biases['bfc1']

    # Dropout 2:
    x_mfcc_dropout_2 = dropout(x_mfcc_2, dropout_prob, is_training)

    # Layer 3: second FC layer
    x_mfcc_3 = tf.matmul(x_mfcc_dropout_2, weights['wfc2']) + biases['bfc2']

    # Dropout 3:
    x_mfcc_dropout_3 = dropout(x_mfcc_3, dropout_prob, is_training)

    # Layer 4: third FC layer
    x_mfcc_output = tf.matmul(x_mfcc_dropout_3, weights['wfc3']) + biases['bfc3']

    return x_mfcc_output


def convSpeechModelC(x_mel_in, x_mfcc_in, x_zcr_in, x_rmse_in, dropout_prob=None, is_training=False):
    """Low latency conv model using the MFCC combined with the ZCR and RMSE in a single audio fingerprint"""

    # ======================================================
    # Setup the parameters for the model
    # ======================================================

    # MFCC with ZCR and RMSE input size (20 MFCCs, 2 features, 2 delta features)
    t_size = 122
    f_size = 24

    # Parameters for Conv layer 1 filter
    filter_size_t = t_size
    filter_size_f = 4
    filter_count = 256
    filter_stride_t = 1
    filter_stride_f = 4

    # Paramaters for FC layers
    fc_output_channels_1 = 128
    fc_output_channels_2 = 128
    fc_output_channels_3 = cfg.N_CLASSES

    # Number of elements in the first FC layer
    fc_element_count = int(filter_count *
                           int(1 + (t_size - filter_size_t) / filter_stride_t) *
                           int(1 + (f_size - filter_size_f) / filter_stride_f))

    # ======================================================
    # Setup dictionaries containing weights and biases
    # ======================================================

    weights = {
        'wconv1': weight_variable([filter_size_t, filter_size_f, 1, filter_count], 'wconv1'),
        'wfc1': weight_variable([fc_element_count, fc_output_channels_1], 'wfc1'),
        'wfc2': weight_variable([fc_output_channels_1, fc_output_channels_2], 'wfc2'),
        'wfc3': weight_variable([fc_output_channels_2, fc_output_channels_3], 'wfc3'),
    }
    biases = {
        'bconv1': bias_variable([filter_count], 'bconv1'),
        'bfc1': bias_variable([fc_output_channels_1], 'bfc1'),
        'bfc2': bias_variable([fc_output_channels_2], 'bfc2'),
        'bfc3': bias_variable([fc_output_channels_3], 'bfc3'),
    }

    # ======================================================
    # Model definition and calculations
    # ======================================================

    # Calculate deltaZCR and deltaRMSE
    x_zcr_delta = tf_diff_axis(x_zcr_in)
    x_rmse_delta = tf_diff_axis(x_rmse_in)


    # Combine MFCC with ZCR, RMSE, deltaZCR and deltaRMSE

    x_mfcc_in, x_zcr_in, x_rmse_in, x_zcr_delta, x_rmse_delta



    # Normalize (L2) along the time axis
    x_mfcc_in_norm = tf.nn.l2_normalize(x_mfcc_in, 1, epsilon=1e-8)

    # Reshape input to [audio file number, time size, freq size, channel]
    x_mfcc_rs = tf.reshape(x_mfcc_in_norm, [-1, t_size, f_size, 1])

    # Layer 1: first Conv layer, BiasAdd and ReLU
    x_mfcc_1 = tf.nn.relu(conv2d(x_mfcc_rs, weights['wconv1'],
                                 sx=filter_stride_t,
                                 sy=filter_stride_f) + biases['bconv1'])

    # Dropout 1:
    x_mfcc_dropout_1 = dropout(x_mfcc_1, dropout_prob, is_training)

    # Flatten layers
    x_mfcc_1_rs = tf.reshape(x_mfcc_dropout_1, [-1, fc_element_count])

    # Layer 2: first FC layer
    x_mfcc_2 = tf.matmul(x_mfcc_1_rs, weights['wfc1']) + biases['bfc1']

    # Dropout 2:
    x_mfcc_dropout_2 = dropout(x_mfcc_2, dropout_prob, is_training)

    # Layer 3: second FC layer
    x_mfcc_3 = tf.matmul(x_mfcc_dropout_2, weights['wfc2']) + biases['bfc2']

    # Dropout 3:
    x_mfcc_dropout_3 = dropout(x_mfcc_3, dropout_prob, is_training)

    # Layer 4: third FC layer
    x_mfcc_output = tf.matmul(x_mfcc_dropout_3, weights['wfc3']) + biases['bfc3']

    return x_mfcc_output


def convSpeechModelD(x_mel_in, x_mfcc_in, x_zcr_in, x_rmse_in, dropout_prob=None, is_training=False):
    """Low latency conv model using the MFCC combined with the ZCR and RMSE in FC layer"""

    # ======================================================
    # Setup the parameters for the model
    # ======================================================

    return