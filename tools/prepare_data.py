"""Downloads the training dataset and removes bad samples.
"""

import csv
import os
import urllib.request
import tarfile
import glob


DATA_URL = 'http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz'
TRAIN_DIR = '../dataset/train/audio/'

FILE_BAD = 'bad_samples.txt'


def maybe_download(data_url, dest_directory):
  """Download and extract data set tar file.
  """
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    print('Downloading %s ...' % filename)
    filepath, _ = urllib.request.urlretrieve(data_url, filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    print('Successfully unzipped %s' % filename)


def remove_bad(f_bad, train_dir):
  """Deletes bad samples in the dataset.
  """
  num_bad = 0
  with open(f_bad, 'r') as fp:
    for wav in csv.reader(fp, delimiter=','):
      try:
        os.remove(train_dir + wav[0])
        num_bad += 1
      except FileNotFoundError:
        pass
  print('bad_training_samples removed: %d' % num_bad)

  wav_paths = glob.glob(os.path.join(train_dir, '*', '*nohash*.wav'))
  print('num_training_samples = %d' % len(wav_paths))


maybe_download(DATA_URL, TRAIN_DIR)
remove_bad(FILE_BAD, TRAIN_DIR)
