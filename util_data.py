"""Data related functions, to build the work and character dictionaries,
  split the datasets and augment the audio.
  The functions which_set() and split_datasets() are based on
  https://github.com/tensorflow/tensorflow/blob/master/tensorflow/
  examples/speech_commands/input_data.py
"""

import urllib
import tarfile
import csv
import collections
import hashlib
import os.path
import re

import scipy.io.wavfile
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
from tensorflow.python.util import compat


SILENCE_CLASS = '9'
SILENCE_WORD = 'silence'
UNKNOWN_WORD = 'unknown'
# the length of all-zero silence wave, can be any number
SILENCE_LENGTH = 16000
# 16-bit PCM  -32768  +32767  int16
INT15_SCALE = np.power(2, 15)

BG_NOISE_FOLDER = '_background_noise_'


# sha1 hash, 160 bits, different speaker possibly same hash
# value 2 ** 27 - 1 chosen by tensorflow/examples/speech_commands
MAX_SPEAKERS = 2 ** 27 - 1  # ~134M


def speaker_hash_mod(speaker):
  """Hashes speaker ids.
    Hash: acts as shuffling.
  """
  speaker_hash = hashlib.sha1(compat.as_bytes(speaker)).hexdigest()
  speaker_hash = int(speaker_hash, 16) % (MAX_SPEAKERS + 1)
  return speaker_hash


def set_divider(data_dir, key_words, num_folds):
  """Sets the markers for dividing dataset into folds for
    training/ validation split.
    Each fold has equal numbers of key word samples (may has different
    numbers of speakers)
  """
  reg = re.compile('.*/[^/]+/(.*)_nohash_.*.wav')

  speakers = []
  for w in key_words:
    for wav in gfile.Glob(os.path.join(data_dir, w, '*nohash*.wav')):
      speaker = reg.search(wav).groups()[0].lower()
      speakers.append(speaker_hash_mod(speaker))

  speakers.sort()
  total = len(speakers)
  size = total // num_folds
  divider = [0] * (num_folds + 1)
  div_idx = [0] * (num_folds + 1)
  for i in range(num_folds):
    divider[i] = speakers[i * size]
    div_idx[i] = speakers.index(divider[i])
  # +1, for range [divider[fold_i],  divider[fold_i + 1]) used in which_set()
  divider[-1] = speakers[-1] + 1
  div_idx[-1] = total - 1

  fold_speakers = [len(set(speakers[div_idx[i]:div_idx[i + 1]]))
                   for i in range(num_folds)]
  tf.logging.debug('%d, dataset divider: %s', total, divider)
  tf.logging.debug('num_speakers_per_fold: %s', fold_speakers)
  return divider


def which_set(speaker, divider, fold_i):
  """Determines which data partition (training/ validation) the file belongs to.
    Training /validation split does not persist, when adding new speakers.
    More equal dividing than which_set_longterm()
  """
  speaker_hash = speaker_hash_mod(speaker)
  if speaker_hash >= divider[fold_i] and speaker_hash < divider[fold_i + 1]:
    result = 'validation'
  else:
    result = 'training'
  return result


def which_set_longterm(speaker, valida_pct, test_pct, fold_idx):
  """Determines which data partition the file belongs to.
    Training /validation /test split persists, when adding new speakers.
  """
  speaker_pct = speaker_hash_mod(speaker) / MAX_SPEAKERS
  if ((speaker_pct < valida_pct * (fold_idx + 1) and
       speaker_pct >= valida_pct * fold_idx)):
    result = 'validation'
  elif (speaker_pct < (test_pct + valida_pct) * (fold_idx + 1) and
        speaker_pct >= (test_pct + valida_pct) * fold_idx):
    result = 'testing'
  else:
    result = 'training'
  return result


def split_datasets(data_dir, word_dict, num_folds, fold_idx):
  """Split known words (including silence) and unknown words into training
    and validation datasets respectively.
  """
  modes = ['training', 'validation']
  knowns = {m: [] for m in modes}
  unknowns = {m: [] for m in modes}
  word_excluded = set()
  reg = re.compile('.*/([^/]+)/(.*)_nohash_(.*).wav')
  # to find the most common known word
  known_counter = {m: collections.Counter() for m in modes}

  key_words = word_dict.key_words
  divider = set_divider(data_dir, key_words, num_folds)
  for wav in gfile.Glob(os.path.join(data_dir, '*', '*nohash*.wav')):
    groups = reg.search(wav).groups()
    word = groups[0].lower()
    speaker = groups[1].lower()
    mode = which_set(speaker, divider, fold_idx)

    indices = word_dict.word_to_indices(word)
    if indices:
      if word in key_words:
        knowns[mode].append([wav, indices])
        known_counter[mode][word] += 1
      else:
        unknowns[mode].append([wav, indices])
    else:
      word_excluded.add(word)
  print('words not in word_map.txt:', word_excluded)

  # make an all-zero silence wave
  silence_dir = os.path.join(data_dir, SILENCE_CLASS)
  if not os.path.exists(silence_dir):
    os.makedirs(silence_dir)
  silence_0 = os.path.join(silence_dir, '%s_0.wav' % SILENCE_WORD)
  encode_audio(np.zeros([SILENCE_LENGTH]), silence_0)

  for mode in modes:
    silence_indices = word_dict.word_to_indices(SILENCE_CLASS)
    silence_size = known_counter[mode].most_common(1)[0][1]
    knowns[mode] += [[silence_0, silence_indices]] * silence_size

  return knowns, unknowns


def map_chars(file_chars, chars=None):
  """Creates character-index mapping. The mapping needs to be constant for
    training and inference.
  """
  if not os.path.exists(file_chars):
    tf.logging.info('WARNING!!!! regenerating %s', file_chars)
    idx_to_char = {i + 1: c for i, c in enumerate(chars)}
    # 0 is not used, dense to sparse array
    idx_to_char[0] = ''
    # null label
    idx_to_char[len(idx_to_char)] = '_'

    with gfile.GFile(file_chars, 'w') as fp:
      for i, c in idx_to_char.items():
        fp.write('%d,%s\n' % (i, c))
  else:
    with gfile.GFile(file_chars, 'r') as fp:
      reader = csv.reader(fp, delimiter=',')
      idx_to_char = {int(i): c for i, c in reader}

  char_to_idx = {c: i for i, c in idx_to_char.items()}
  return idx_to_char, char_to_idx


class WordDict():
  """ WordDict class that provides word/character related methods, including
    converting original words to modified version (modi) (e.g. removing
    unpronounced characters), creating/keeping word and character dictionaries,
    calculating distribution of words in a dataset or a batch, converting
    words into character indices for training and the reverse for predictions.
  """
  def __init__(self, file_words, file_chars, num_key_words):
    self._word_to_modi = {}
    self._modi_to_word = {}
    self._word_to_modi[SILENCE_CLASS] = SILENCE_CLASS
    self._modi_to_word[SILENCE_CLASS] = SILENCE_WORD
    self._all_words = []
    with gfile.GFile(file_words, 'r') as fp:
      reader = csv.reader(fp, delimiter=',')
      for row in reader:
        if not row[0].startswith('#'):
          org = row[0]
          train_w = row[1]
          self._all_words.append(org)
          self._word_to_modi[org] = train_w
          self._modi_to_word[train_w] = org
    self._all_words += [UNKNOWN_WORD, SILENCE_WORD]

    train_modis = list(self._word_to_modi.values())
    self._num_word_classes = len(train_modis)
    self._max_label_length = max([len(w) for w in train_modis])

    chars = list(set(''.join(train_modis)))
    self._idx_to_char, self._char_to_idx = map_chars(file_chars, chars=chars)
    self._num_char_classes = len(self._idx_to_char)

    # modi_to_target dictionary
    self._key_words = self._all_words[:num_key_words]
    self._modi_to_target = {}
    for modi, word in self._modi_to_word.items():
      if word in self._key_words + [SILENCE_WORD]:
        self._modi_to_target[modi] = word

    # word_to_target dictionary
    self._word_to_target = {}
    for word in self._all_words:
      if word in self._key_words + [SILENCE_WORD]:
        self._word_to_target[word] = word
      else:
        self._word_to_target[word] = UNKNOWN_WORD

  def word_distro(self, words, msg='', verbose=False):
    """Calculates the distribution of words and characters.
    """
    w_cnt = collections.Counter()
    for w in words:
      w_cnt[self.indices_to_modi(w)] += 1

    tf.logging.debug('[%s] %d, %d, %s', msg, len(w_cnt),
                     sum(w_cnt.values()), w_cnt)

    if verbose:
      c_cnt = collections.Counter()
      for w in list(w_cnt.keys()):
        for c in list(w):
          c_cnt[c] += w_cnt[w]
      tf.logging.debug('[%s] %d, %s', msg, len(c_cnt), c_cnt)

      w_sum = sum(w_cnt.values())
      for key in w_cnt.keys():
        w_cnt[key] = round(128 * w_cnt[key] / w_sum, 1)
      tf.logging.debug('[%s] %d, %s', msg, len(w_cnt), w_cnt)

    return w_cnt

  @property
  def num_classes(self):
    return self._num_word_classes, self._num_char_classes

  @property
  def max_label_length(self):
    return self._max_label_length

  @property
  def key_words(self):
    return self._key_words

  @property
  def all_words(self):
    return self._all_words

  def word_to_indices(self, word):
    """Converts original words to modified version(modi), then to character
      indices for training.
    """
    try:
      modi = self._word_to_modi[word]
      return [self._char_to_idx[c] for c in modi]
    except KeyError:
      return None

  def indices_to_modi(self, indices):
    """Converts character indices to modified version of words, used for
      training/ validation debugging.
    """
    modi = ''.join([self._idx_to_char[i] for i in indices])
    return modi

  def modi_to_word(self, modi):
    """Converts modified version of words to the original words."""
    try:
      word = self._modi_to_word[modi]
    except KeyError:
      word = UNKNOWN_WORD
    return word

  def indices_to_submit(self, indices):
    """Converts character indices to key words (including silence) and unknown
      words for submission.
    """
    modi = ''.join([self._idx_to_char[i] for i in indices])
    try:
      word = self._modi_to_target[modi]
    except KeyError:
      word = UNKNOWN_WORD
    return word

  def word_to_submit(self, word):
    """Converts non-key works to 'unknown' for submission."""
    return self._word_to_target[word]


def maybe_download(data_url, dest_directory):
  """Downloads and extracts data set tar file.
  """
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = data_url.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    tf.logging.info('Downloading %s ...', filename)
    filepath, _ = urllib.request.urlretrieve(data_url, filepath)
    tarfile.open(filepath, 'r:gz').extractall(dest_directory)
    tf.logging.info('Successfully unzipped %s', filename)


def decode_audio(wav_file, target_length=-1):
  """Decodes audio wave file.
    audio_new.shape:[target_length]
  """
  _, audio = scipy.io.wavfile.read(wav_file)
  audio = audio.astype(np.float32, copy=False)
  # keep 0 not shifted. Not: (audio + 0.5) * 2 / (INT15_SCALE * 2 - 1)
  audio /= INT15_SCALE
  audio_length = len(audio)

  if target_length != -1:
    if audio_length < target_length:
      audio_new = np.zeros(target_length, dtype=np.float32)
      start = (target_length - audio_length) // 2
      audio_new[start:start + audio_length] = audio
    else:
      start = (audio_length - target_length) // 2
      audio_new = audio[start:start + target_length]
  else:
    audio_new = audio
  return audio_new


def encode_audio(audio, wav_file, sample_rate=16000):
  """Encodes to audio wave file.
    audio.shape:[target_length]
  """
  audio = audio * INT15_SCALE
  audio = np.clip(audio, -INT15_SCALE, INT15_SCALE - 1)
  audio = audio.reshape([-1]).astype(np.int16, copy=False)
  scipy.io.wavfile.write(wav_file, sample_rate, audio)
  tf.logging.debug('encode_audio: saving to %s', wav_file)


def load_bg_audios(data_dir, audio_length, bg_noise_folder=BG_NOISE_FOLDER):
  """Loads background noise files.
  """
  path = os.path.join(data_dir, bg_noise_folder)
  if os.path.exists(path):
    wav_files = gfile.Glob(os.path.join(path, '*.wav'))
    bg_audios = [decode_audio(w) for w in wav_files]
  else:
    print('bg noise path %s does not exist' % path)
    bg_audios = [np.zeros([audio_length])]
  return bg_audios


def select_bg_audio(bg_audios, target_length, volume_max=1):
  """Chooses a background noise from bg_audios list and returns a copy
    of the target length and scaled volume.
    bg_audios.type: list
  """
  index = np.random.randint(len(bg_audios))
  audio = bg_audios[index]
  # randint: len(audio) - target_length + 1, high exclusive
  start = np.random.randint(0, len(audio) - target_length + 1)
  audio = audio[start:start + target_length]
  volume = np.random.uniform(0, volume_max)
  scaled = audio * volume
  return scaled


def augment_audio(audio, bg_audios, target_length, fg_strech, bg_noise_prob,
                  bg_nsr):
  """Augments the foreground audio using time stretching, random padding
    and mixing with the background noise.
    audio.shape: [audio_length]
  """
  # time stretching
  if np.random.uniform(0, 1) < bg_noise_prob:
    audio_length = len(audio)
    step = np.random.uniform(1 / fg_strech, fg_strech)
    audio = np.interp(np.arange(0, audio_length, step),
                      np.arange(0, audio_length), audio)

  # fix length, random padding
  audio_length = len(audio)
  if audio_length <= target_length:
    pad_total = target_length - audio_length
    pad_bf = np.random.randint(0, pad_total + 1)
    fg_audio = np.pad(audio, [pad_bf, pad_total - pad_bf], mode='constant')
  else:
    start = (audio_length - target_length) // 2
    fg_audio = audio[start:start + target_length]

  # bg_noise_prob = 0: add no noise
  # bg_noise_prob != 0, bg_nsr = 0: only add noise to silence file
  if np.random.uniform(0, 1) < bg_noise_prob:
    fg_max = np.abs(fg_audio).max()
    if fg_max == 0:
      # the silence foreground audio
      volumn_max = 1
    else:
      volumn_max = fg_max * bg_nsr
    bg_noise = select_bg_audio(bg_audios, target_length, volumn_max)
  else:
    bg_noise = 0

  fg_audio += bg_noise
  fg_audio = np.clip(fg_audio, -1.0, 1.0)

  return fg_audio
