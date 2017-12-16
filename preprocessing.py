#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Signal preprocessing functions for tensorflow speech recognition.
Note that the rank of the tensor is included to allow for both cases of single files
and batches of files.
Note that these functions must all be run in a tf session.
"""

import tensorflow as tf
from tensorflow.contrib import signal
import config as cfg


def windower(arr, window=128, hop_length=32, rank=1):
    """Windower function that divides an array into fixed size windows.
       Similar to signal.frame but is faster to execute."""
    overlap = window - hop_length
    length = arr.shape[rank - 1]
    indexer = tf.range(window)[None, :] + hop_length * tf.range((length - overlap) / hop_length)[:, None]
    return tf.gather(arr, indexer, axis=rank - 1)


def tf_diff_axis(arr):
    """Equivalent of np.diff on final axis"""
    return arr[..., 1:] - arr[..., :-1]


def zero_crossing(arr, rank=1):
    """Calculates the zero crossing rate"""
    if rank == 1:
        nzc = tf.cast(tf.count_nonzero(tf_diff_axis(tf.sign(arr))), tf.float32)
    else:
        nzc = tf.cast(tf.count_nonzero(tf_diff_axis(tf.sign(arr)), axis=rank - 1), tf.float32)

    arrlen = tf.cast(arr.shape[rank - 1], tf.float32)
    return tf.divide(nzc, arrlen, name='zcr')


def rms_energy(arr, rank=1, maxamps=1.0):
    """Calculates the RMS energy of the wave"""
    if rank == 1:
        return tf.sqrt(tf.reduce_mean(tf.square(arr / maxamps)), name='rmse')
    else:
        return tf.sqrt(tf.reduce_mean(tf.square(arr / maxamps), axis=rank - 1), name='rmse')


def signalProcessBatch(signals, noise_factor=0.1, noise_frac=0.2, window=512,
                       maxamps=1.0, sr=16000, num_mel_bins=64, num_mfccs=13):
    """Function to perform all the DSP preprocessing and feature extraction.
       Returns the MFCCs, Log Mel spectrum, ZCR and RMSE.
       Works on a batch of num_files files.
       - Input signals : tensor of shape [num_files, samples]
       - Output        : tuple ([num_files, num_windows, num_mfccs],
                                [num_files, num_windows, num_mel_bins],
                                [num_files, num_windows],
                                [num_files, num_windows])"""

    # Get number of signal files
    num_files = tf.shape(signals)[0]

    # Select random noise samples
    idx = tf.random_uniform((num_files,), 0, cfg.NOISE_MATRIX.shape[0], dtype=tf.int32)
    noise = tf.cast(tf.gather(cfg.NOISE_MATRIX, idx), tf.float32)
    nf = tf.cast(tf.greater(tf.random_uniform([num_files, 1]), noise_frac), tf.float32)

    # Add noise to signal with a certain noise factor
    signals = signals + noise_factor * maxamps * noise * nf

    # Window the audio signals
    hop_length = window / 4
    signals32 = tf.cast(signals, tf.float32)
    signals_w = windower(signals32, window=window, hop_length=hop_length, rank=2)

    # Calculate Zero Crossing Rate and RMSE
    zcr = zero_crossing(signals_w, rank=3)
    rmse = rms_energy(signals_w, rank=3, maxamps=maxamps)

    # Calculate the Short Time Fourier Transform
    stfts = signal.stft(signals32, frame_length=window, frame_step=hop_length, fft_length=window)
    magnitude_spectrograms = tf.abs(stfts)

    # Define Mel space
    num_spectrogram_bins = magnitude_spectrograms.shape[-1].value
    lower_edge_hertz = 80.0
    upper_edge_hertz = 7600.0
    mel_weight_mat = signal.linear_to_mel_weight_matrix(num_mel_bins,
                                                        num_spectrogram_bins,
                                                        sr,
                                                        lower_edge_hertz,
                                                        upper_edge_hertz)

    # Calculate the Mel spectrogram and set its shape
    mel_spectrograms = tf.tensordot(magnitude_spectrograms, mel_weight_mat, 1)
    mel_spectrograms.set_shape(magnitude_spectrograms.shape[:-1].concatenate(mel_weight_mat.shape[-1:]))

    # Calculate log of the spectrogram
    log_offset = 1e-8
    log_mel_spectrograms = tf.log(mel_spectrograms + log_offset, name='log_mel_spectrograms')

    # Calcuate the MFCCs
    mfccs = signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :num_mfccs]
    mfccs = tf.identity(mfccs, name='mfccs')

    return mfccs, log_mel_spectrograms, zcr, rmse
