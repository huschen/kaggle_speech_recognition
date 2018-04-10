"""Training-related functions and classes, to run evaluation, record the stats,
   save ckpts and run test set inference.
"""

import os
import re
import logging
import random
from time import gmtime, strftime

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import tqdm

import audio_dataset
import util_data


def train_init(fold_idx, start_ckpt, log_dir=None):
  """Sets random seeds and sets up file logging.
  """
  if start_ckpt:
    groups = re.search('.*_([0-9]*_[0-9]*).ckpt-([0-9]*)',
                       start_ckpt).groups()
    t_stamp = groups[0]
    init_step = int(groups[1])
  else:
    if dbg_basic():
      # verify the basic code, without training
      t_stamp = '0_0'
    else:
      t_stamp = strftime("%m%d_%H%M", gmtime())
    init_step = 0

  tf.set_random_seed(fold_idx)
  # seed for training set
  np.random.seed(fold_idx + init_step)
  # seed for validation set
  random.seed(fold_idx)

  tf.logging.set_verbosity(tf.logging.DEBUG)
  if log_dir:
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    log_file = os.path.join(log_dir, 'training_%s.log' % t_stamp)
    if os.path.exists(log_file) and dbg_basic():
      os.remove(log_file)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
    tf.logging._logger.addHandler(file_handler)

  # for printing confusion matrix
  np.set_printoptions(linewidth=160)

  return t_stamp, init_step


def tf_session(gmem_dynamic=False):
  """Creates a TensorFlow session,
    with GPU memory dynamically growing or all allocated.
  """
  if gmem_dynamic:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
  else:
    sess = tf.Session()
  return sess


def dbg_basic():
  """Mode to verify the basic code, without training.
    e.g. when there is no GPUs available.
  """
  return not tf.test.is_gpu_available()


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


class PerfRecorder():
  """Performance recorder that records wrongly predicted files, calculates
    confusion matrix and accuracy.
  """
  def __init__(self, word_dict, debug_dir, t_stamp, step):
    self._wdict = word_dict
    self._wrongs = ''
    self._debug_dir = debug_dir
    self._time_step = '%s-%s' % (t_stamp, step)
    self._step = step

    all_words = word_dict.all_words
    num_words = len(all_words)
    self._word_idx = {w: i for i, w in enumerate(all_words)}
    self._conf_matrx = np.zeros([num_words, num_words], dtype='int')

  def record(self, edist, pred, labels, wavs, score):
    predit_wrong = edist != 0
    reg = re.compile('.*/([^/]+/.*.wav)')
    word_idx = self._word_idx
    for i, p in enumerate(pred):
      true_modi = self._wdict.indices_to_modi(labels[i])
      pred_modi = self._wdict.indices_to_modi(p)
      if predit_wrong[i]:
        self._wrongs += '%s,%s,%s,%.3f,%d\n' % (
            reg.search(wavs[i]).groups()[0],
            true_modi, pred_modi, score[i], edist[i])

      true_word = self._wdict.modi_to_word(true_modi)
      pred_word = self._wdict.modi_to_word(pred_modi)
      self._conf_matrx[word_idx[true_word]][word_idx[pred_word]] += 1

  def report(self):
    debug_path = os.path.join(
        self._debug_dir, 'v_wrongs_%s.txt' % self._time_step)
    with gfile.GFile(debug_path, 'w') as fp:
      fp.write(self._wrongs)

    # all the key words are placed at the top of map_words.txt and
    # silence should be placed at the end of all_words list.
    total, num_correct, key_key, key_as_unkn, unkn_as_key, unkn_unkn = \
        coarse_stats(self._conf_matrx, len(self._wdict.key_words), -1)
    submit_correct = num_correct + unkn_unkn

    msg_dbg = '\n\n [%s]: total = %d, wrong_pred = %d, wrong_submit = %d, '\
              'acc_beam, acc_submit: [%.2f%%, %.2f%%] \n  wrong: ' \
              'key_key = %d, key_as_unknown = %d, unknown_as_key = %d\n' \
              % (self._step, total, total - num_correct,
                 total - submit_correct, num_correct * 100 / total,
                 submit_correct * 100 / total,
                 key_key, key_as_unkn, unkn_as_key)

    debug_path = os.path.join(
        self._debug_dir, 'v_confusion_matrix_%s.txt' % self._time_step)
    with gfile.GFile(debug_path, 'w') as fp:
      all_words = self._wdict.all_words
      fp.write('true>pred %s\n' % ','.join(['%3s' % w[:3] for w in all_words]))
      for i in range(len(all_words)):
        fp.write('%s\t%s\n' % (all_words[i], self._conf_matrx[i]))
      fp.write(msg_dbg)

    tf.logging.info(msg_dbg)


class Evaluator():
  """Validation evaluator that runs validations, records the stats, also
    saves the histories of prediction details for chosen samples (optional).
  """
  def __init__(self, dataset, mode, graph, sess, batch_size, word_dict,
               debug_dir, t_stamp, audio_length, sample_wavs=None):
    self._dataset = dataset
    self._mode = mode
    self._graph = graph
    self._sess = sess
    self._batch_size = batch_size
    self._wdict = word_dict
    self._debug_dir = debug_dir
    self._t_stamp = t_stamp
    self._samples, self._hisf = self.samples_init(audio_length, sample_wavs)

  def samples_init(self, audio_length, sample_wavs):
    samples = []
    hisf = []
    if sample_wavs:
      for wav in sample_wavs:
        try:
          samples.append(util_data.decode_audio(wav, audio_length))
          f_name = 'hist_%s_%s.txt' % (wav.split('/')[-2], self._t_stamp)
          hisf.append(os.path.join(self._debug_dir, f_name))
        except IOError:
          pass
    return samples, hisf

  def samples_eval(self, train_step):
    if self._samples:
      raw_preds, preds, scores, probs = self._graph.pred_dbg(self._sess,
                                                             self._samples)
      for i, f in enumerate(self._hisf):
        str_prob = ' '.join(['%.2f' % p for p in probs[i]])
        msg = '%d, %s %s [%s], %s %s, %.3f' % (
            train_step, raw_preds[i],
            self._wdict.indices_to_modi(raw_preds[i]), str_prob, preds[i],
            self._wdict.indices_to_modi(preds[i]), scores[i])
        with gfile.GFile(f, 'a') as fp:
          fp.write(msg + '\n')

  def run_evaluation(self, train_step):
    set_size = self._dataset.size
    avg_acc_beam = 0
    avg_loss = 0
    sum_score = 0
    wrong_submits = 0
    sum_edist = 0
    for _ in range(0, set_size, self._batch_size):
      data, labels, _ = self._dataset.next_batch(self._batch_size)

      loss, acc_beam, edist, predicts, \
          scores = self._graph.validation(self._sess, data, labels)

      data_size = len(labels)
      scale = data_size * 100 / set_size
      avg_acc_beam += acc_beam * scale
      avg_loss += loss * scale
      sum_score += scores.sum()
      sum_edist += edist.sum() / set_size

      for i in range(data_size):
        true_submit = self._wdict.indices_to_submit(labels[i])
        pred_submit = self._wdict.indices_to_submit(predicts[i])
        wrong_submits += int(pred_submit != true_submit)

    acc_submit = (1 - wrong_submits / set_size) * 100
    tf.logging.info('Step %d: [%s] acc_beam = %.2f%%, acc_submit = %.2f%%, '
                    'loss = %.4f, sum_edist = %.4f, confidence = %.3f',
                    train_step, self._mode, avg_acc_beam,
                    acc_submit, avg_loss / 100, sum_edist,
                    sum_score / set_size)
    # return a list of score/scores for Ckpter.recorder.
    return (avg_acc_beam, acc_submit)

  def stats_validation(self, step):
    set_size = self._dataset.size

    recorder = PerfRecorder(self._wdict, self._debug_dir, self._t_stamp, step)
    avg_acc_beam = 0
    for _ in range(0, set_size, self._batch_size):
      data, labels, wavs = self._dataset.next_batch(self._batch_size)

      _, acc_beam, edist, predicts, \
          scores = self._graph.validation(self._sess, data, labels)

      data_size = len(labels)
      avg_acc_beam += (acc_beam * data_size) / set_size
      recorder.record(edist, predicts, labels, wavs, scores)

    recorder.report()


class Ckpter():
  """Check point worker that saves and restores check points
    also returns the ckpts of the top-n performance.
  """
  def __init__(self, sess, ckpt_dir, name, t_stamp, max_ckpt=20, num_best=1):
    self._sess = sess
    self._saver = tf.train.Saver(tf.global_variables(), max_to_keep=max_ckpt)
    self._ckpt_path = os.path.join(ckpt_dir, '%s_%s.ckpt' % (name, t_stamp))
    self._ckpt_dir = ckpt_dir
    self._name = name
    self._t_stamp = t_stamp
    # circular buffer, [step, [score_list]] pair
    self._saver_scores = [[0, [0]]] * max_ckpt
    self._saver_idx = 0
    self._max_ckpt = max_ckpt
    self._num_best = num_best

  @property
  def saver(self):
    return self._saver

  def save(self, step):
    self._saver.save(self._sess, self._ckpt_path, global_step=step)
    tf.logging.info('Saved to "%s-%d"', self._ckpt_path, step)

  def restore(self, ckpt):
    checkpoint_path = os.path.join(self._ckpt_dir, ckpt)
    self._saver.restore(self._sess, checkpoint_path)

  def record(self, step, score):
    self.save(step)
    idx = self._saver_idx % self._max_ckpt
    self._saver_scores[idx] = [step, score]
    self._saver_idx += 1

  def get_best(self, skip_gap=200):
    best_ckpts = []
    best_steps = [-skip_gap]
    scores = sorted(self._saver_scores, key=lambda x: sum(x[1]), reverse=True)

    for i, (step, score) in enumerate(scores[:self._num_best]):
      ckpt = '%s_%s.ckpt-%d' % (self._name, self._t_stamp, step)
      score_str = ','.join(['%.2f%%' % s for s in list(score)])
      do_add = all([abs(step - s) >= skip_gap for s in best_steps])
      if do_add:
        tf.logging.debug('Best*%d: %s, Score: [%s]', i, ckpt, score_str)
        best_ckpts.append(os.path.join(self._ckpt_dir, ckpt))
        best_steps.append(step)
      else:
        tf.logging.debug('Best %d: %s, Score: [%s]', i, ckpt, score_str)
    return best_ckpts


def infer_testset(graph, sess, audio_length, word_dict, wav_list,
                  result_dir, batch_size, saver, ckpt, mode='submission'):
  """Runs the predictions on the test set, produces submission csv and
    another debug version with more details.
  """
  saver.restore(sess, ckpt)
  groups = re.search('.*_([0-9]*_[0-9]*).ckpt-([0-9]*)', ckpt).groups()
  time_step = '%s-%s' % (groups[0], groups[1])

  csvf = os.path.join(result_dir, '%s_%s.csv' % (mode, time_step))
  csvf_dbg = os.path.join(result_dir, '%s_dbg_%s.csv' % (mode, time_step))
  tf.logging.info('saving to %s', csvf)

  dataset = audio_dataset.AudioInferSet(wav_list, audio_length)

  with gfile.GFile(csvf + 'tmp', 'w') as fp, \
          gfile.GFile(csvf_dbg + 'tmp', 'w') as fdbg:
    fp.write('fname,label\n')
    fdbg.write('fname,label,score,pred\n')

    for _ in tqdm.tqdm(range(0, dataset.size, batch_size)):
      files, audio = dataset.next_batch(batch_size)
      predicts, scores = graph.pred_score(sess, audio)

      for i, pred in enumerate(predicts):
        wav = os.path.basename(files[i])
        word = word_dict.indices_to_submit(pred)
        if mode == 'train':
          true_word = files[i].split('/')[-2]
          wav = '%s_%s' % (true_word, wav)
        fp.write('%s,%s\n' % (wav, word))

        modi = word_dict.indices_to_modi(pred)
        word = word_dict.modi_to_word(modi)
        fdbg.write('%s,%s,%.3f,%s\n' % (wav, word, scores[i], modi))

  os.rename(csvf + 'tmp', csvf)
  os.rename(csvf_dbg + 'tmp', csvf_dbg)
