#!/usr/bin/env python

import os
import re
import hashlib

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M


def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.

    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.

    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.

    Args:
        filename: File path of the data sample.
        validation_percentage: How much of the data set to use for validation.
        testing_percentage: How much of the data set to use for testing.

    Returns:
        String, one of 'training', 'validation', or 'testing'.
    """

    base_name = os.path.basename(filename)

    # Ignore anything after '_nohash_' in the file name
    hash_name = re.sub(r'_nohash_.*$', '', base_name)

    # Hash file name and generate probability of being train, test or val
    hash_name_hashed = hashlib.sha1(hash_name).hexdigest()
    hash_val = int(hash_name_hashed, 16) % (MAX_NUM_WAVS_PER_CLASS + 1)
    percentage_hash = hash_val * (100.0 / MAX_NUM_WAVS_PER_CLASS)

    if percentage_hash < validation_percentage:
        result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
        result = 'testing'
    else:
        result = 'training'
    return result
