#!/bin/bash
#
# Scripts to download the training set from http://download.tensorflow.org, and
# download the test set using Kaggle API (https://github.com/Kaggle/kaggle-api)
#

set -e

sdir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$sdir"

python prepare_data.py

mkdir -p "$sdir"/../dataset
cd "$sdir"/../dataset

if [ ! -f test.7z ]; then
  if type kaggle > /dev/null 2>&1; then
    kaggle competitions download -c tensorflow-speech-recognition-challenge -f test.7z -p .
    mv tensorflow-speech-recognition-challenge/test.7z .
    rm -r tensorflow-speech-recognition-challenge
    7z x test.7z
  fi
fi

echo num_test_samples = `find test/audio/ -name '*.wav' | wc -l`
