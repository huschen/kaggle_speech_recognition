"""Classes and functions to provide data batches for training and inferring.
"""
import os
import random
import signal
import multiprocessing

import numpy as np
import tensorflow as tf

import util_data


# audio process pool, multi-processes to speed up audio augmentation.
# non-deterministic.
_audio_process_pool = None
# enable multiprocessing debugging
# multiprocessing.log_to_stderr().setLevel(multiprocessing.SUBDEBUG)

_bg_audios = []


def init_audio_process(bg_audios=None):
  """Initialization function for creating audio processes.
    - Disables SIGINT in child processes and leave the main process to turn off
    the daemon child processes.
    - Updates _bg_audios, so that AudioAugmentor in child processes can use
    bg audio directly without getting it from the parent process repeatedly.
    - Resets random seed, so that different child processes have different
    random seeds, not the same one as the parent process.
  """
  def sig_handler(*unused):
    """doing_nothing sig_handler"""
    return None
  signal.signal(signal.SIGINT, sig_handler)

  if bg_audios:
    global _bg_audios
    _bg_audios = bg_audios

    identity = multiprocessing.current_process()._identity
    np.random.seed(sum(identity) ** 2)

  print('audio_pid=%s, bg_audios=%d' % (os.getpid(), len(_bg_audios)))


def get_audio_pool(initargs):
  """Returns the audio process pool. Creates the pool if it does not exist.
  """
  global _audio_process_pool
  if _audio_process_pool is None:
    _audio_process_pool = multiprocessing.Pool(initializer=init_audio_process,
                                               initargs=initargs)
  return _audio_process_pool


def load_datasets(data_url, data_dir, word_dict, unknown_pct, num_folds,
                  fold_idx, audio_length, fg_pad, bg_noise_prob, bg_nsr):
  """Downloads the data and creates train and validation datasets.
  """
  util_data.maybe_download(data_url, data_dir)
  bg_audios = util_data.load_bg_audios(data_dir, audio_length)
  max_label_length = word_dict.max_label_length

  knowns, unknowns = util_data.split_datasets(data_dir, word_dict,
                                              num_folds, fold_idx)
  # for training, using two separate datasets (known and unknown),
  # so more unknowns words can be used for training
  mode = 'training'
  dataset_train = AudioNoisySet(
      [knowns[mode], unknowns[mode]], [1 - unknown_pct, unknown_pct],
      audio_length, max_label_length, bg_audios, fg_pad, bg_noise_prob, bg_nsr)

  # get the distribution of known and unknown words. re-zip (wav, word) pair.
  word_dict.word_distro(list(zip(*knowns[mode]))[1], msg=mode)
  word_dict.word_distro(list(zip(*unknowns[mode]))[1], msg=mode)
  tf.logging.debug('')

  # for validation, using one dataset (containing both knowns and unknowns)
  mode = 'validation'
  random.shuffle(unknowns[mode])
  unknown_size = int(len(knowns[mode]) * unknown_pct / (1 - unknown_pct))
  knowns[mode] += unknowns[mode][:unknown_size]
  # shuffle once, fixed order each validation.
  random.shuffle(knowns[mode])
  dataset_valida = AudioCleanSet(knowns[mode], audio_length, max_label_length)
  word_dict.word_distro(list(zip(*knowns[mode]))[1], msg=mode, verbose=True)

  return dataset_train, dataset_valida


class DataSet(object):
  """Basic DataSet class.
  """
  def __init__(self, data):
    self._d_size = len(data)
    self._data = data
    self._epochs_completed = 0
    self._i_in_epoch = 0

  @property
  def size(self):
    """Size of the dataset."""
    return self._d_size

  def nb_raw(self, batch_size, batch_wrap=False, shuffle=False):
    """Return the next `batch_size` examples from this data set.
    """
    start = self._i_in_epoch
    if self._epochs_completed == 0 and start == 0 and shuffle:
      np.random.shuffle(self._data)

    if start + batch_size >= self._d_size:
      # Finished epoch
      self._epochs_completed += 1
      self._i_in_epoch = 0
      data_batch = self._data[start:self._d_size]
      if shuffle:
        np.random.shuffle(self._data)

      if batch_wrap:
        # Start next epoch
        self._i_in_epoch = batch_size - (self._d_size - start)
        end = self._i_in_epoch
        data_new_part = self._data[0:end]
        data_batch.extend(data_new_part)
        # data_batch=np.hstack([data_batch, data_new_part])
      return data_batch
    else:
      self._i_in_epoch += batch_size
      end = self._i_in_epoch
      return self._data[start:end]


class DataSetMulti(DataSet):
  """DataSetMulti class, supporting multiple data sources.
  """
  def __init__(self, datas, ratios):
    self._data = datas
    self._ratios = ratios
    num_datas = len(datas)
    self._d_size = [len(data) for data in datas]
    self._epochs_completed = [0] * num_datas
    self._i_in_epoch = [0] * num_datas

  @property
  def size(self):
    """Size of the dataset. It is not the actual size of the data sources in
      the dataset, but the equivalent to that of the basic DataSet mode.
      Calculated using the first data source and its percentage.
    """
    return int(self._d_size[0] / self._ratios[0])

  @property
  def list_size(self):
    """Actual sizes of the data sources in this dataset.
    """
    return self._d_size

  def nb_raw_single(self, batch_size, i_data, batch_wrap=False, shuffle=False):
    """Return the next `batch_size` examples from single data source.
    """
    start = self._i_in_epoch[i_data]
    if self._epochs_completed[i_data] == 0 and start == 0 and shuffle:
      np.random.shuffle(self._data[i_data])

    if start + batch_size >= self._d_size[i_data]:
      # Finished epoch
      self._epochs_completed[i_data] += 1
      self._i_in_epoch[i_data] = 0
      data_batch = self._data[i_data][start:self._d_size[i_data]]
      if shuffle:
        np.random.shuffle(self._data[i_data])

      if batch_wrap:
        # Start next epoch
        self._i_in_epoch[i_data] = batch_size - (self._d_size[i_data] - start)
        end = self._i_in_epoch[i_data]
        data_new_part = self._data[i_data][0:end]
        data_batch.extend(data_new_part)
        # data_batch=np.hstack([data_batch, data_new_part])
      return data_batch
    else:
      self._i_in_epoch[i_data] += batch_size
      end = self._i_in_epoch[i_data]
      return self._data[i_data][start:end]

  def nb_raw(self, batch_size, batch_wrap=False, shuffle=True):
    """Return the next `batch_size` examples from multiple data sources.
    """
    sizes = [int(batch_size * r) for r in self._ratios]
    sizes[0] = batch_size - sum(sizes[1:])
    data = []
    for i, size in enumerate(sizes):
      data += self.nb_raw_single(size, i, batch_wrap, shuffle)
    np.random.shuffle(data)
    return data


class AudioDecoder(object):
  """Class to process clean audio files, used in validation.
  """
  def __init__(self, audio_length):
    self._audio_length = audio_length

  def __call__(self, wav_file):
    audio = util_data.decode_audio(wav_file, self._audio_length)
    audio = audio.reshape(1, -1)
    return audio


class AudioAugmentor(AudioDecoder):
  """Class to process and augment audio files,
    used in training by audio child processes.
  """
  def __init__(self, audio_length, fg_pad, bg_noise_prob, bg_nsr):
    self._audio_length = audio_length
    self._fg_strech = audio_length / (audio_length - fg_pad)
    self._bg_noise_prob = bg_noise_prob
    self._bg_nsr = bg_nsr

  def __call__(self, wav_file):
    audio = util_data.decode_audio(wav_file)
    # use global _bg_audios in the child process, to save passing big data
    # from the parent process to child processes repeatedly.
    audio = util_data.augment_audio(audio, _bg_audios, self._audio_length,
                                    self._fg_strech, self._bg_noise_prob,
                                    self._bg_nsr)
    audio = audio.reshape(1, -1)
    return audio


class AudioCleanSet(DataSet):
  """Dataset of clean audio and their labels,
    used in validation by audio child processes.
  """
  def __init__(self, data, audio_length, max_label_length,
               proc_func=AudioDecoder):
    super().__init__(data)
    self._max_label_length = max_label_length
    self._pool = get_audio_pool([])
    self._func = proc_func(audio_length)
    self._nb_raw = super().nb_raw

  def next_batch(self, batch_size, batch_wrap=False, shuffle=False):
    data_batch = self._nb_raw(batch_size, batch_wrap, shuffle)
    batch_size = len(data_batch)

    wav_list, label_list = zip(*data_batch)
    audios_list = self._pool.map(self._func, wav_list)
    audios = np.vstack(audios_list)

    labels = np.zeros([batch_size, self._max_label_length], dtype=np.int)
    for i, label in enumerate(label_list):
      labels[i, 0:len(label)] = label

    wavs = np.array(wav_list, dtype=np.object)
    return audios, labels, wavs


class AudioNoisySet(DataSetMulti, AudioCleanSet):
  """Dataset of noisy audio and their labels, used in training.
  """
  def __init__(self, datas, ratios, audio_length, max_label_length, bg_audios,
               fg_pad, bg_noise_prob, bg_nsr, proc_func=AudioAugmentor):
    super().__init__(datas, ratios)
    self._max_label_length = max_label_length
    self._pool = get_audio_pool([bg_audios])
    self._func = proc_func(audio_length, fg_pad, bg_noise_prob, bg_nsr)
    self._nb_raw = super().nb_raw


class AudioInferSet(DataSet):
  """Dataset of clean audio, used in inference.
  """
  def __init__(self, data, audio_length, proc_func=AudioDecoder):
    super().__init__(data)
    self._pool = get_audio_pool([])
    self._func = proc_func(audio_length)

  def next_batch(self, batch_size, batch_wrap=False, shuffle=False):
    wav_list = super().nb_raw(batch_size, batch_wrap, shuffle)
    audios_list = self._pool.map(self._func, wav_list)
    audios = np.vstack(audios_list)
    return wav_list, audios
