#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Training script for speech recognition neural net.
"""

import os
import sys
import tensorflow as tf

import time
from datetime import datetime, timedelta

import config as cfg
from dataload import load_data, load_batch, load_config
from preprocessing import signalProcessBatch
from models import buildModel


def main(_):

    # Load parameters from yaml file specified in argv
    paramfilename = sys.argv[1]
    modelname, param = load_config(paramfilename)
    n_classes = cfg.N_CLASSES

    melspec_size = param['melspec_shape'][0] * param['melspec_shape'][1]
    mfcc_size = param['mfcc_shape'][0] * param['mfcc_shape'][1]

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        # Placeholders for signals preprocessing inputs
        X_data = tf.placeholder(tf.float32, [None, cfg.SAMRATE], name='X_data')

        noise_factor = tf.placeholder(tf.float32, shape=(), name='noise_factor')
        noise_frac = tf.placeholder(tf.float32, shape=(), name='noise_frac')

        # Define the audio features
        x_mfcc, x_mel, x_zcr, x_rmse = signalProcessBatch(X_data,
                                                          noise_factor=noise_factor,
                                                          noise_frac=noise_frac,
                                                          window=param['window'],
                                                          maxamps=cfg.MAXAMPS, sr=cfg.SAMRATE,
                                                          num_mel_bins=param['num_mel_bins'],
                                                          num_mfccs=param['num_mfccs'])

        # Placeholder variables output (1-hot vectors of size n_classes)
        y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
        y_true_class = tf.argmax(y_true, 1, name='y_true_class')

        # Dropout keep probability and training flag
        dropout_prob = tf.placeholder(tf.float32, shape=(), name='dropout_prob')
        is_training = tf.placeholder(tf.bool, name="is_training")

        # Prediction from model
        model = buildModel(modelname)
        y_pred = model(x_mel, x_mfcc, x_zcr, x_rmse,
                       dropout_prob=dropout_prob, is_training=is_training)

        # Cross entropy loss function with softmax then takes mean
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true))
        tf.summary.scalar('loss', loss)

        # Train and backprop gradients function
        optimizer = tf.train.AdamOptimizer(learning_rate=param['learning_rate']).minimize(loss)

        # Evaluation and accuracy
        y_pred_class = tf.argmax(y_pred, 1, name='y_pred_class')
        correct_prediction = tf.equal(y_pred_class, y_true_class)
        confusion_matrix = tf.confusion_matrix(y_true_class, y_pred_class, num_classes=n_classes)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        # Merge all summaries
        merged = tf.summary.merge_all()

        # Saver for checkpoints
        saver = tf.train.Saver(tf.global_variables())

        # Set path to summary logs and checkpoints
        now = datetime.now()
        logs_path = os.path.join(cfg.OUT_DIR, now.strftime("%Y%m%d-%H%M%S"), 'summaries')

        # Create summary writers
        train_writer = tf.summary.FileWriter(os.path.join(logs_path, 'train'), graph=sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(logs_path, 'test'), graph=sess.graph)

        # Initialize variables
        tf.global_variables_initializer().run()

        # Main training section
        start_time = time.time()
        msg = "\n====================\nStarting training...\n===================="
        tf.logging.info(msg)

        # Load the audio file info dataframe
        df = load_data(cfg.DATA_DIR)

        # Log
        msg = "\nModel: {}\nParam File: {}\nIterations: {}"
        tf.logging.info(msg.format(modelname, paramfilename, param['num_iterations']))

        tf.logging.info(" Begin iterations...")
        for i in xrange(param['num_iterations']):

            # Get the training batch
            X_train, y_true_batch = load_batch(df, cfg.DATA_DIR,
                                               batch_size=param['batch_size'],
                                               silence_size=param['silence_size'],
                                               label='train',
                                               random=True, seed=None,
                                               w=param['unknown_weight'],
                                               samples=cfg.SAMRATE)

            # Preprocess the training batch
            x_mfcc_batch, x_mel_batch, x_zcr_batch, x_rmse_batch = sess.run(
                [x_mfcc, x_mel, x_zcr, x_rmse],
                feed_dict={X_data: X_train,
                           noise_factor: param['noise_factor_value'],
                           noise_frac: param['noise_frac_value']})

            # Training optimization
            sess.run(optimizer, feed_dict={x_mel: x_mel_batch,
                                           x_mfcc: x_mfcc_batch,
                                           x_zcr: x_zcr_batch,
                                           x_rmse: x_rmse_batch,
                                           y_true: y_true_batch,
                                           dropout_prob: param['dropout_prob_value'],
                                           is_training: True})

            # Checkpoint save and validation step
            if ((i + 1) % param['checkpoint_step'] == 0) or (i == param['num_iterations'] - 1):

                # Checkpoint
                checkpoint_path = os.path.join(logs_path, "{}-{}.ckpt".format(modelname, paramfilename[:-4]))
                msg = " Saving checkpoint to: {}-{}"
                tf.logging.info(msg.format(checkpoint_path, i + 1))
                saver.save(sess, checkpoint_path, global_step=i + 1)

                # Load the validation batches
                val_batch_size = 100
                total_val_accuracy = 0
                total_conf_matrix = None
                val_set_size = 6700
                for j in xrange(0, val_set_size, val_batch_size - param['silence_size']):
                    X_val, y_true_val = load_batch(df, cfg.DATA_DIR,
                                                   batch_size=val_batch_size,
                                                   silence_size=param['silence_size'],
                                                   label='val',
                                                   random=False, seed=j,
                                                   w=1.0, samples=cfg.SAMRATE)

                    # Preprocess the validation batch
                    x_mfcc_val, x_mel_val, x_zcr_val, x_rmse_val = sess.run(
                        [x_mfcc, x_mel, x_zcr, x_rmse],
                        feed_dict={X_data: X_val,
                                   noise_factor: 0.0,
                                   noise_frac: 0.0})

                    # Validation summary
                    val_summary, loss_val, acc_val, conf_matrix = sess.run(
                        [merged, loss, accuracy, confusion_matrix],
                        feed_dict={x_mel: x_mel_val,
                                   x_mfcc: x_mfcc_val,
                                   x_zcr: x_zcr_val,
                                   x_rmse: x_rmse_val,
                                   y_true: y_true_val,
                                   dropout_prob: 1.0,
                                   is_training: False})
                    total_val_accuracy += (acc_val * val_batch_size) / val_set_size
                    if total_conf_matrix is None:
                        total_conf_matrix = conf_matrix
                    else:
                        total_conf_matrix += conf_matrix

                msg = " Confusion Matrix:\n {}"
                tf.logging.info(msg.format(total_conf_matrix))
                msg = " VALIDATION ACC: {:6f}, (N = {})"
                tf.logging.info(msg.format(total_val_accuracy, val_set_size))

            # Display step
            if (i == 0) or ((i + 1) % param['display_step'] == 0) or (i == param['num_iterations'] - 1):
                # Training summary, loss and accuracy
                train_summary, loss_train, acc_train = sess.run(
                    [merged, loss, accuracy],
                    feed_dict={x_mel: x_mel_batch,
                               x_mfcc: x_mfcc_batch,
                               x_zcr: x_zcr_batch,
                               x_rmse: x_rmse_batch,
                               y_true: y_true_batch,
                               dropout_prob: 1.0,
                               is_training: False})
                train_writer.add_summary(train_summary, i + 1)

                # Display message
                msg = "  OPTIMIZE STEP: {:6d}, LOSS, {:.6f}, ACC: {:.6f}"
                tf.logging.info(msg.format(i + 1, loss_train, acc_train))

                # Check if loss is below minimum
                if loss_train < param['min_loss']:
                    msg = " Min loss acheived: {}"
                    tf.logging.info(msg.format(loss_train))
                    break

        # Load the testing batches
        test_batch_size = 100
        total_test_accuracy = 0
        total_conf_matrix = None
        test_set_size = 6700
        for j in xrange(0, test_set_size, test_batch_size - param['silence_size']):
            X_test, y_true_test = load_batch(df, cfg.DATA_DIR,
                                             batch_size=test_batch_size,
                                             silence_size=param['silence_size'],
                                             label='test',
                                             random=False, seed=j,
                                             w=1.0, samples=cfg.SAMRATE)

            # Preprocess the testing batch
            x_mfcc_test, x_mel_test, x_zcr_test, x_rmse_test = sess.run(
                [x_mfcc, x_mel, x_zcr, x_rmse],
                feed_dict={X_data: X_test,
                           noise_factor: 0.0,
                           noise_frac: 0.0})

            # Testing summary
            test_summary, loss_test, acc_test, conf_matrix = sess.run(
                [merged, loss, accuracy, confusion_matrix],
                feed_dict={x_mel: x_mel_test,
                           x_mfcc: x_mfcc_test,
                           x_zcr: x_zcr_test,
                           x_rmse: x_rmse_test,
                           y_true: y_true_test,
                           dropout_prob: 1.0,
                           is_training: False})
            test_writer.add_summary(test_summary, i + 1)

            total_test_accuracy += (acc_test * test_batch_size) / test_set_size
            if total_conf_matrix is None:
                total_conf_matrix = conf_matrix
            else:
                total_conf_matrix += conf_matrix

        msg = " Confusion Matrix:\n {}"
        tf.logging.info(msg.format(total_conf_matrix))
        msg = " TESTING ACC: {:6f}, (N = {})"
        tf.logging.info(msg.format(total_test_accuracy, test_set_size))

        # End-time
        end_time = time.time()
        msg = " Time usage: {}"
        tf.logging.info(msg.format(timedelta(seconds=int(round(end_time - start_time)))))


if __name__ == '__main__':
    tf.app.run(main=main)
