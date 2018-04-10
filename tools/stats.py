"""Abstrac stats from training and testing results (Draft!!).
"""
import csv
import glob
import collections
import sys
import os
import shutil

import numpy as np


if len(sys.argv) == 3:
  DATA_DIR = sys.argv[1]
  INPUT_DIR = sys.argv[2]
else:
  DATA_DIR = '../dataset/'
  INPUT_DIR = '../submissions/avg_0504_1400-768/'

print(DATA_DIR, INPUT_DIR)


KEY_WORDS = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop',
             'go']

KNOWN_WORDS = KEY_WORDS + ['zero', 'one', 'two', 'three', 'four', 'five',
                           'six', 'seven', 'eight', 'nine', 'house', 'happy',
                           'bird', 'wow', 'bed', 'dog', 'cat', 'marvin',
                           'sheila', 'tree', 'unknown', 'silence']

np.set_printoptions(linewidth=160)


def process_csvs(stats_type, dir_level='/', input_dir=INPUT_DIR):
  f_dbg = '%s_dbg_*.csv' % stats_type
  # dir_level = '/*/' or '/', input_dir/*/f_dbg or input_dir/f_dbg
  csvs = glob.glob(input_dir + dir_level + f_dbg)
  if len(csvs) == 0:
    return None
  print('\n%s,input_csvs: ' % stats_type, len(csvs), csvs)
  is_train = stats_type == 'train'

  known_cntr = collections.Counter()
  unkn_cntr = collections.Counter()
  wav_dct = collections.defaultdict(list)
  score_dct = collections.defaultdict(collections.Counter)
  hit_dct = collections.defaultdict(collections.Counter)
  if is_train:
    recall_dct = collections.defaultdict(collections.Counter)
  else:
    recall_dct = None

  for f in csvs:
    with open(f, 'r') as fp:
      reader = csv.reader(fp, delimiter=',')
      _ = next(reader)
      for line in reader:
        wav, word, score, pred = line[:4]
        if is_train:
          true_word = wav.split('_')[0]
          recall_dct[true_word][word] += 1
        if word == 'unknown':
          word = pred
          unkn_cntr[word] += 1
        else:
          known_cntr[word] += 1
        wav_dct[word].append([wav, score])
        score_dct[wav][word] += float(score)
        hit_dct[wav][word] += 1

  if len(csvs) == 1:
    print(len(known_cntr.keys()), known_cntr.most_common())
    print(len(unkn_cntr.keys()), unkn_cntr.most_common(30), '\n')

  return csvs, known_cntr, unkn_cntr, wav_dct, score_dct, hit_dct, recall_dct


def coarse_stats(conf_matrx, div, end):
  """Abstracts coarse-grained stats from the confusion matrix.
    The wrong predictions are grouped into four categories, e.g.,
    key_word_as_another_key_word, key_as_unknown_word, unknown_word_as_key,
    unknown_as_another_unknown.
  """
  total = conf_matrx.sum()
  num_correct = np.trace(conf_matrx)
  key_key = conf_matrx[:div, :div].sum() - np.trace(conf_matrx[:div, :div])
  key_as_unkn = conf_matrx[:div, div:end].sum()
  unkn_as_key = conf_matrx[div:end, :div].sum()
  unkn_unkn = conf_matrx[div:end, div:end].sum() - \
      np.trace(conf_matrx[div:end, div:end])
  return total, num_correct, key_key, key_as_unkn, unkn_as_key, unkn_unkn


def result_train(dir_level='/', data_dir=DATA_DIR, cp_file=False):
  stats_type = 'train'
  try:
    csvs, _, _, _, score_dct, \
        hit_dct, recall_dct = process_csvs(stats_type, dir_level)
  except TypeError:
    return

  if len(csvs) == 1:
    word_idx = {w: i for i, w in enumerate(KNOWN_WORDS)}
    num_knowns = len(KNOWN_WORDS)
    conf_matrx = np.zeros([num_knowns, num_knowns], dtype='int')

    for true_word in recall_dct.keys():
      for pred, num in recall_dct[true_word].most_common():
        conf_matrx[word_idx[true_word], word_idx[pred]] = num
    print('true>pred%s' % ','.join(['%4s' % w[:4] for w in KNOWN_WORDS]))
    for i in range(num_knowns):
      print('%s\t%s' % (KNOWN_WORDS[i], conf_matrx[i]))

    total, correct, key_key, key_as_unkn, unkn_as_key, unkn_unkn = \
        coarse_stats(conf_matrx, len(KEY_WORDS), -1)
    key_as_unkn_unkn = conf_matrx[:len(KEY_WORDS), -2].sum()
    wrong = total - correct

    print('wrong_key_key=%d, key_as_unknown=%d (%d), unknown_as_key=%d' % (
        key_key, key_as_unkn, key_as_unkn_unkn, unkn_as_key))
    print('accuracy=[%.3f%%, %.3f%%], total=%d, wrong=%d, wrong_submit=%d' % (
        correct * 100.0 / total, (correct + unkn_unkn) * 100.0 / total,
        total, wrong, wrong - unkn_unkn))

  else:  # len(csvs) != 1:
    if cp_file:
      dest_dir = os.path.join(data_dir, 'tmp_stats_%s' % stats_type)
      shutil.rmtree(dest_dir, ignore_errors=True)
      os.makedirs(dest_dir)
      print('\nSaving interestingly-predicted training data in %s' % dest_dir)

    for wav in hit_dct.keys():
      word, score = score_dct[wav].most_common(1)[0]
      _, hit = hit_dct[wav].most_common(1)[0]
      if (hit == 1 and word in KEY_WORDS) or (word == 'silence'):
        print(wav, score_dct[wav].most_common())
        if cp_file:
          dst = os.path.join(dest_dir, '%s_[%s]_%s' % (wav, score, word))
          wav = wav.replace('_', '/', 1)
          src = os.path.join(data_dir, 'train', 'audio', *wav.split('/'))
          os.symlink(os.path.relpath(src, dest_dir), dst)


def copy_wavs(wavs, word, cntr, src_dir, dst_dir, thresh=100):
  """Helper function used by result_test(), to create symlink of wavs.
  """
  for wav, score in wavs:
    if float(score) < thresh:
      src = os.path.join(src_dir, wav)
      dst = os.path.join(dst_dir, '%05d_%s_[%s]_%s' % (cntr, word, score, wav))
      os.symlink(os.path.relpath(src, dst_dir), dst)


def result_test(dir_level='/', data_dir=DATA_DIR, cp_file=False, cp_num=30):
  stats_type = 'submission'
  try:
    csvs, known_cntr, unkn_cntr, wav_dct, score_dct, \
        hit_dct, _ = process_csvs(stats_type, dir_level)
  except TypeError:
    return

  if cp_file and len(csvs) == 1:
    dest_dir = os.path.join(data_dir, 'tmp_stats_%s' % stats_type)
    shutil.rmtree(dest_dir, ignore_errors=True)
    os.makedirs(dest_dir)
    print('\nSaving test samples in %s' % dest_dir)
    test_dir = os.path.join(data_dir, 'test', 'audio')

    for word, counter in known_cntr.most_common():
      if word in KEY_WORDS + ['four', 'zero', 'seven']:
        copy_wavs(wav_dct[word][0:cp_num], word, counter, test_dir, dest_dir)

    for word, counter in unkn_cntr.most_common(7):
      copy_wavs(wav_dct[word][0:cp_num], word, counter, test_dir, dest_dir)

  wavs = [('clip_19d1d3485.wav', 'follow'), ('clip_67fd58f37.wav', 'forward'),
          ('clip_f71bb6134.wav', 'backward'), ('clip_235c60ad6.wav', 'learn')]

  for _, (wav, label) in enumerate(wavs):
    print('%s, %s: %s, %s' % (wav, label, score_dct[wav], hit_dct[wav]))


result_train(dir_level='/')
# result_train(dir_level='/*/', cp_file=True)
result_test(dir_level='/')
result_test(dir_level='/*/')
