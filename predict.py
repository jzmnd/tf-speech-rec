#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to run prediction on test files and output submission .csv
"""

import os
import sys
import tensorflow as tf

from datetime import datetime

import config as cfg
from dataload import load_test_batch


def export_csv(clipnames, predictions, outfile):
    """Exports a csv file in the format required by Kaggle"""
    header = "fname,label\n"
    with open(outfile, 'w') as f:
        f.write(header)
        for i, clipname in enumerate(clipnames):
            f.write("{},{}\n".format(clipname, predictions[i]))


def main(_):

    modeldir = sys.argv[1]
    modelname = sys.argv[2]

    modelpath = os.path.join(cfg.OUT_DIR, modeldir)
    modelfull = os.path.join(modelpath, 'summaries', modelname)

    with tf.Session() as sess:
        tf.logging.set_verbosity(tf.logging.INFO)

        # Add ops to save and restore all the variables from meta graph
        saver = tf.train.import_meta_graph("{}.meta".format(modelfull))

        # Restore variables from disk
        saver.restore(sess, modelfull)
        graph = tf.get_default_graph()

        # Get required placeholder tensor names
        y_pred_class = graph.get_tensor_by_name('y_pred_class:0')
        X_data = graph.get_tensor_by_name('X_data:0')
        dropout_prob = graph.get_tensor_by_name('dropout_prob:0')
        noise_factor = graph.get_tensor_by_name('noise_factor:0')
        noise_frac = graph.get_tensor_by_name('noise_frac:0')
        is_training = graph.get_tensor_by_name('is_training:0')

        msg = " Model restored: {}"
        tf.logging.info(msg.format(modelname))

        # Test batch data
        num_tests = 158538
        batch_size = 200
        num_batches = num_tests / batch_size + 1
        num_rem = num_tests % batch_size

        # Lists of results
        predictions = []
        filenames_list = []

        # Loop through all test files in batches
        for i in xrange(num_batches):

            if i == (num_batches - 1):
                b = num_rem
            else:
                b = batch_size

            # Load the batch
            X_test_values, X_list = load_test_batch(cfg.DATA_DIR,
                                                    idx=i * batch_size,
                                                    batch_size=b,
                                                    samples=cfg.SAMRATE)

            # Run prediction
            y_pred_class_result = sess.run(y_pred_class,
                                           feed_dict={X_data: X_test_values,
                                                      dropout_prob: 1.0,
                                                      noise_factor: 0.0,
                                                      noise_frac: 0.0,
                                                      is_training: False})

            msg = "   BATCH: {:4d} /{:4d}   DONE"
            tf.logging.info(msg.format(i + 1, num_batches))

            # Add results to lists
            filenames_list.extend(X_list)
            predictions.extend([cfg.NUM2LABEL[k] for k in y_pred_class_result])

        # Export results to a .csv file in modelpath directory
        now = datetime.now()
        outfile = os.path.join(modelpath, "output_{}.csv".format(now.strftime("%Y%m%d-%H%M%S")))
        export_csv(filenames_list, predictions, outfile)


if __name__ == '__main__':
    tf.app.run(main=main)
