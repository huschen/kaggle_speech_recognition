"""Implementation of the Conv-LSTM-CTC network model including:
  audio_to_spectrogram converting,
  3-layer convolutional residual net (each layer: conv, res, res, max_pool) ,
  backward LSTM and two fully-connected layers.
"""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import io_ops
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio


# only support Mono channel for simplicity, audio.shape: [n_samples, 1]
def decode_audio(wav_file, desired_samples=-1):
  """Builds a TensorFlow graph to decode wave file.
    wav_file: scalar, dtype: string
    audio.shape: [n_samples, 1]
  """
  with tf.name_scope('decode_audio'):
    file_content = io_ops.read_file(wav_file)
    audio, sample_rate = contrib_audio.decode_wav(
        file_content, desired_channels=1, desired_samples=desired_samples)
  return audio, sample_rate


def audio_to_spectrogram(audio, sample_rate, frame_size=480.0,
                         frame_stride=160.0, num_mel_bins=40.0, debug=False):
  """Builds a TensorFlow graph to convert audio to spectrogram.
    audio.shape: [batch_size, sample_length]
    magnit_spectro.shape:[batch_size, num_time_frame, num_spectrogram_bins]
    scale_log.shape: [batch_size, num_time_frame, num_mel_bins]
  """

  with tf.name_scope('audio_to_spectrogram'):
    magnit_spectro = tf.abs(tf.contrib.signal.stft(
        audio, frame_length=frame_size, frame_step=frame_stride,
        fft_length=frame_size))
    num_spectro_bins = magnit_spectro.shape[-1].value

    mel_weight_matrix = tf.contrib.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectro_bins, sample_rate,
        lower_edge_hertz=20.0, upper_edge_hertz=4000.0)

    mel_spectro = tf.tensordot(
        magnit_spectro, mel_weight_matrix, 1)
    mel_spectro.set_shape(magnit_spectro.shape[:-1].concatenate(
        mel_weight_matrix.shape[-1:]))
    v_max = tf.reduce_max(mel_spectro, axis=[1, 2], keep_dims=True)
    v_min = tf.reduce_min(mel_spectro, axis=[1, 2], keep_dims=True)
    is_zero = tf.to_float(tf.equal(v_max - v_min, 0))
    scale_mel = (mel_spectro - v_min) / (v_max - v_min + is_zero)

    epsilon = 0.001
    log_spectro = tf.log(scale_mel + epsilon)
    v_min = np.log(epsilon)
    v_max = np.log(epsilon + 1)
    scale_log = (log_spectro - v_min) / (v_max - v_min)
  if debug:
    return magnit_spectro, mel_spectro, scale_mel, scale_log
  else:
    return scale_log


class AudioSpectro():
  """Builds an Audio to Spectrogram model.
  """
  def __init__(self, audio_length, sample_rate, frame_size, frame_stride,
               num_mel_bins):
    tf.logging.info('AudioSpectro parameters: %s',
                    [audio_length, frame_size, frame_stride, num_mel_bins])

    self._audio = tf.placeholder(tf.float32, [None, audio_length],
                                 name='audio')
    self._spectro_data = audio_to_spectrogram(
        self._audio, sample_rate, frame_size, frame_stride,
        num_mel_bins, debug=True)

  def evaluate(self, sess, audio):
    # spectro_data: magnit_spectro, mel_spectro, scale_mel, scale_log
    return sess.run(self._spectro_data, feed_dict={self._audio: audio})


def layer_convbr(net, conv1_kernel, bn_train, pad='SAME', relu=True):
  """Network layer containing conv, batch_norm and relu.
  """
  weights = tf.Variable(
      tf.truncated_normal(conv1_kernel, stddev=0.01), name='kernel')
  net = tf.nn.conv2d(net, weights, [1, 1, 1, 1], pad)

  net = tf.layers.batch_normalization(net, training=bn_train)
  if relu:
    net = tf.nn.relu(net)
  return net


def layer_residual(net, conv_kernel, bn_train):
  """Residual layer containing two convs (with batch_norm) and relu.
  """
  org = net
  num_channels = net.shape[-1].value
  assert (num_channels == conv_kernel[-1]), "in_channels == out_channels!"

  net = layer_convbr(net, conv_kernel, bn_train)
  net = layer_convbr(net, conv_kernel, bn_train, relu=False)

  net = net + org
  net = tf.nn.relu(net)
  return net


def layer_convres_maxpool(net, conv_kernel, res_kernel, pool_kernel, bn_train):
  """Network layer containing conv, residual and relu.
  """
  net = layer_convbr(net, conv_kernel, bn_train, pad='VALID')
  net = layer_residual(net, res_kernel, bn_train)
  net = layer_residual(net, res_kernel, bn_train)
  tf.logging.debug('resnet_output.shape = %s', net.shape)

  net = tf.nn.max_pool(net, pool_kernel, pool_kernel, 'SAME')
  tf.logging.debug('maxpool_out.shape = %s', net.shape)
  return net


def layer_lstm(net, rnn_seq_length, rnn_num_hid, keep_prob, bn_train):
  """Backward LSTM layer.
  """
  # calculate depth, it is possible rnn_seq_length != net.shape[1]
  depth = np.prod(net.shape.as_list()[1:]) // rnn_seq_length
  net = tf.reshape(net, [-1, rnn_seq_length, depth])
  tf.logging.debug('lstm_input.shape = %s', net.shape)

  net = tf.reverse(net, axis=[1])
  fw_cell = tf.nn.rnn_cell.LSTMCell(rnn_num_hid, use_peepholes=True)
  fw_drop_cell = tf.nn.rnn_cell.DropoutWrapper(
      fw_cell, input_keep_prob=keep_prob, state_keep_prob=keep_prob,
      variational_recurrent=True, dtype=tf.float32, input_size=depth)

  output, _ = tf.nn.dynamic_rnn(fw_drop_cell, net, dtype=tf.float32,
                                time_major=False, scope='rnn')
  out_reverse = tf.reverse(output, axis=[1])
  net = tf.layers.batch_normalization(out_reverse, training=bn_train)
  return net


def layer_fcbr(net, num_classes, bn_train, relu=True):
  """Network layer containing feed-forward, batch_norm and leaky_relu.
  """
  num_units = net.shape[-1].value
  weights = tf.Variable(
      tf.truncated_normal([num_units, num_classes], stddev=0.01),
      name='kernel')
  net = tf.matmul(net, weights)

  net = tf.layers.batch_normalization(net, training=bn_train)
  if relu:
    net = tf.nn.leaky_relu(net, alpha=0.75)
  return net


def nn_conv_lstm(nn_input, num_classes, keep_prob, bn_train):
  """Builds the main graph.
    nn_input.shape:[batch_size, num_time_frame, num_mel_bins]
  """
  tf.logging.info('nn_input.shape = %s, is_training=%s',
                  nn_input.shape, keep_prob != 1)
  dbg_layers = []
  dbg_embeddings = []
  dbg_layers.append(tf.expand_dims(nn_input, -1))

  with tf.name_scope('bn'):
    net = tf.layers.batch_normalization(nn_input, training=bn_train)
    net = tf.expand_dims(net, -1)

  with tf.name_scope('cnn'):
    # cnn_layer 1: conv, res, res, max_pool
    layer1_channel = 64
    conv1_kernel = [5, 3, 1, layer1_channel]
    res1_kernel = [3, 3, layer1_channel, layer1_channel]
    pool_kernel = [1, 2, 2, 1]
    net = layer_convres_maxpool(net, conv1_kernel, res1_kernel,
                                pool_kernel, bn_train)
    dbg_layers.append(net)

    # cnn_layer 2: conv, res, res, max_pool
    layer2_channel = 128
    conv2_kernel = [3, 3, layer1_channel, layer2_channel]
    res2_kernel = [3, 3, layer2_channel, layer2_channel]
    pool_kernel = [1, 2, 2, 1]
    net = layer_convres_maxpool(net, conv2_kernel, res2_kernel,
                                pool_kernel, bn_train)
    dbg_layers.append(net)

    # cnn_layer 3: conv, res, res, max_pool
    layer3_channel = 256
    conv3_kernel = [3, 3, layer2_channel, layer3_channel]
    res3_kernel = [3, 3, layer3_channel, layer3_channel]
    pool_kernel = [1, 2, 2, 1]
    net = layer_convres_maxpool(net, conv3_kernel, res3_kernel,
                                pool_kernel, bn_train)
    dbg_layers.append(net)
    tf.logging.info('cnn_out.shape = %s', net.shape)

  with tf.name_scope('rnn'):
    rnn_seq_length = net.shape[1].value
    rnn_num_hid = 768
    net = layer_lstm(net, rnn_seq_length, rnn_num_hid, keep_prob, bn_train)
    dbg_layers.append(net)
    dbg_embeddings.append(net)
  tf.logging.info('lstm_output.shape = %s', net.shape)

  with tf.name_scope('fc'):
    net = tf.reshape(net, [-1, net.shape[-1].value])
    tf.logging.debug('fc_input.shape = %s', net.shape)
    # fc_layer 1
    fc_num_hid = 160
    if keep_prob != 1:
      net = tf.nn.dropout(net, keep_prob)
    net = layer_fcbr(net, fc_num_hid, bn_train)
    dbg_embeddings.append(tf.reshape(net, [-1, rnn_seq_length, fc_num_hid]))
    tf.logging.debug('fc_output.shape = %s', net.shape)

    # fc_layer 2
    if keep_prob != 1:
      net = tf.nn.dropout(net, keep_prob)
    net = layer_fcbr(net, num_classes, bn_train, relu=False)
    tf.logging.debug('fc_output.shape = %s', net.shape)

    logits = tf.reshape(net, [-1, rnn_seq_length, num_classes])
    dbg_embeddings.append(logits)
    tf.logging.info('fc_logits.shape = %s', logits.shape)

  return logits, dbg_layers, dbg_embeddings


def cal_perf(pred, sparse_labels):
  """Helper function to calculate edit distance and accuracy.
  """
  edist = tf.edit_distance(tf.cast(pred[0], tf.int32), sparse_labels,
                           normalize=False)
  acc = tf.reduce_mean(tf.cast(tf.equal(edist, 0), tf.float32))
  return edist, acc


def nn_cost_ctc(logits, labels, label_lengths):
  """Calculates network CTC cost, accuracy, predictions and confidence scores.
  """
  with tf.name_scope('loss_ctc'):
    idx = tf.where(tf.not_equal(labels, 0))
    sparse_labels = tf.SparseTensor(idx, tf.gather_nd(labels, idx),
                                    tf.shape(labels, out_type=tf.int64))
    loss = tf.reduce_mean(tf.nn.ctc_loss(
        sparse_labels, logits, label_lengths,
        preprocess_collapse_repeated=True, time_major=False))
    logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
    probs = tf.nn.softmax(logits)
    max_probs = tf.reduce_max(probs, axis=2)
    raw_preds = tf.argmax(probs, axis=2)

  with tf.name_scope('acc_greedy'):
    pred_greedy, _ = tf.nn.ctc_greedy_decoder(logits_transposed, label_lengths)
    edist_greey, acc_greedy = cal_perf(pred_greedy, sparse_labels)
    edist_greey = tf.reduce_mean(edist_greey)

  with tf.name_scope('acc_beam'):
    pred_beam, prob_scores = tf.nn.ctc_beam_search_decoder(
        logits_transposed, label_lengths, top_paths=2)
    edist, acc_beam = cal_perf(pred_beam, sparse_labels)
    preds = tf.cast(tf.sparse_tensor_to_dense(pred_beam[0]), tf.int32)
    scores = prob_scores[:, 0] - prob_scores[:, 1]

  tf.summary.scalar('ctc_loss', loss)
  tf.summary.scalar('acc_greedy', acc_greedy)
  tf.summary.scalar('edist_greey', edist_greey)
  # tf.summary.scalar('confidence_score', tf.reduce_mean(scores))

  return loss, acc_greedy, acc_beam, edist, preds, scores, raw_preds, max_probs


def nn_optimizer(loss, l2_scale, lr_init, lr_dstep, lr_drate, dbg=False):
  """Network cost optimizer, l2_regularizer, gradient clipping
    and batch_norm update.
  """
  global_step = tf.train.get_or_create_global_step()
  train_variables = tf.trainable_variables()

  sum_l2_loss = 0
  num_params = 0
  with tf.name_scope('regularizer'):
    for var in train_variables:
      if 'kernel' in var.op.name:
        num_params += np.prod(var.shape.as_list())
        v_loss = tf.nn.l2_loss(var)
        if dbg:
          tf.summary.scalar(var.op.name + '/w_l2', v_loss)
        sum_l2_loss += v_loss
    loss += sum_l2_loss * l2_scale
    tf.logging.debug('num_weights_params=%d', num_params)

  with tf.name_scope('optimizer'):
    max_grad_norm = 10.0
    learning_rate = tf.train.exponential_decay(lr_init, global_step, lr_dstep,
                                               lr_drate, staircase=True)
    optimizer = tf.train.AdamOptimizer(learning_rate, epsilon=1e-8)
    grads = tf.gradients(loss, train_variables)
    mod_grads, ctc_global_norm = tf.clip_by_global_norm(grads, max_grad_norm)

    if dbg:
      for var in tf.global_variables():
        if 'batch_normalization' in var.op.name:
          tf.summary.histogram(var.op.name, var)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    tf.logging.debug('optimizer dependencies: %d', len(update_ops))
    with tf.control_dependencies(update_ops):
      train_step = optimizer.apply_gradients(zip(mod_grads, train_variables),
                                             global_step=global_step)

  tf.summary.scalar('learning_rate', learning_rate)
  tf.summary.scalar('l2loss', sum_l2_loss)
  tf.summary.scalar('global_norm', ctc_global_norm)

  tf.logging.debug('l2_scale=%f, max_grad_norm=%f', l2_scale, max_grad_norm)
  return train_step, global_step


class AudioNN():
  """Builds the model graph, for training.
  """
  def __init__(self, FLAGS, num_classes, max_label_length, lr_dstep,
               is_training, log_dir):
    sample_rate = FLAGS.sample_rate
    audio_length = int(sample_rate * FLAGS.target_duration_ms / 1000)
    frame_size = int(sample_rate * FLAGS.frame_size_ms / 1000)
    frame_stride = int(sample_rate * FLAGS.frame_stride_ms / 1000)
    num_mel_bins = FLAGS.num_mel_bins
    lr_init = FLAGS.lr_init
    lr_drate = FLAGS.lr_drate
    l2_scale = FLAGS.l2_scale
    tf.logging.info('AudioNN parameters: %s',
                    [audio_length, frame_size, frame_stride, num_mel_bins])

    self._audio = tf.placeholder(tf.float32, [None, audio_length],
                                 name='audio')
    if is_training:
      self._keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    else:
      self._keep_prob = 1
    self._bn_train = tf.placeholder(tf.bool, name='bn_train')

    # convert audio to spectrogram
    nn_input = audio_to_spectrogram(
        self._audio, sample_rate, frame_size, frame_stride, num_mel_bins)

    # network base
    logits, _, _ = nn_conv_lstm(nn_input, num_classes, self._keep_prob,
                                self._bn_train)

    # ctc_cost function
    label_lengths = tf.fill([tf.shape(nn_input)[0]], max_label_length)
    self._labels = tf.placeholder(tf.int32, [None, max_label_length],
                                  name='label')
    self._loss, self._acc_fgreedy, self._acc_beam, self._edist, \
        self._predicts, self._scores, self._raw_preds, \
        self._max_probs = nn_cost_ctc(logits, self._labels, label_lengths)

    # optimizer
    self._train_step, self._global_step = nn_optimizer(
        self._loss, l2_scale, lr_init, lr_dstep, lr_drate)

    self._summaries = tf.summary.merge_all()
    self._train_writer = tf.summary.FileWriter(log_dir + '/train')
    self._validation_writer = tf.summary.FileWriter(
        log_dir + '/validation', tf.get_default_graph())

  def eval_train(self, sess, audio, labels, keep_prob):
    summary, _, global_step, loss, acc_fgreedy = sess.run(
        [self._summaries, self._train_step, self._global_step,
         self._loss, self._acc_fgreedy],
        feed_dict={
            self._audio: audio,
            self._labels: labels,
            self._keep_prob: keep_prob,
            self._bn_train: True})
    self._train_writer.add_summary(summary, global_step)
    return loss, acc_fgreedy

  def validation(self, sess, audio, labels):
    summary, loss, acc_beam, edist, predicts, scores, global_step = sess.run(
        [self._summaries, self._loss, self._acc_beam,
         self._edist, self._predicts, self._scores, self._global_step],
        feed_dict={
            self._audio: audio,
            self._labels: labels,
            self._keep_prob: 1,
            self._bn_train: False})
    self._validation_writer.add_summary(summary, global_step)
    return loss, acc_beam, edist, predicts, scores

  def pred_score(self, sess, audio):
    predicts, scores = sess.run(
        [self._predicts, self._scores],
        feed_dict={
            self._audio: audio,
            self._keep_prob: 1,
            self._bn_train: False})
    return predicts, scores

  def pred_dbg(self, sess, audio):
    raw_preds, predicts, scores, max_probs = sess.run(
        [self._raw_preds, self._predicts, self._scores, self._max_probs],
        feed_dict={
            self._audio: audio,
            self._keep_prob: 1,
            self._bn_train: False})
    return raw_preds, predicts, scores, max_probs

  def eval_global_step(self, sess):
    return sess.run(self._global_step)


class AudioInferer():
  """Builds the inference graph.
  """
  def __init__(self, FLAGS, num_classes, max_label_length):
    sample_rate = FLAGS.sample_rate
    audio_length = int(sample_rate * FLAGS.target_duration_ms / 1000)
    frame_size = int(sample_rate * FLAGS.frame_size_ms / 1000)
    frame_stride = int(sample_rate * FLAGS.frame_stride_ms / 1000)
    num_mel_bins = FLAGS.num_mel_bins
    tf.logging.info('audios_inference parameters: %s',
                    [audio_length, frame_size, frame_stride, num_mel_bins])

    self._audio = tf.placeholder(tf.float32, [None, audio_length],
                                 name='audio')
    nn_input = audio_to_spectrogram(
        self._audio, sample_rate, frame_size, frame_stride, num_mel_bins)

    self._bn_train = tf.placeholder(tf.bool, name='bn_train')
    logits, self._dbg_layers, self._dbg_embeddings = nn_conv_lstm(
        nn_input, num_classes, keep_prob=1, bn_train=self._bn_train)
    self._prob = tf.nn.softmax(logits)
    self._chars = tf.argmax(self._prob, axis=2)

    label_lengths = tf.fill([tf.shape(nn_input)[0]], max_label_length)
    logits_transposed = tf.transpose(logits, perm=[1, 0, 2])
    pred, self._score_cmp = tf.nn.ctc_beam_search_decoder(
        logits_transposed, label_lengths, top_paths=2)
    self._pred_cmp = [tf.cast(tf.sparse_tensor_to_dense(p), tf.int32)
                      for p in pred]
    self._predicts = self._pred_cmp[0]
    self._scores = self._score_cmp[:, 0] - self._score_cmp[:, 1]

  def pred_score(self, sess, audio):
    predicts, scores = sess.run(
        [self._predicts, self._scores],
        feed_dict={
            self._audio: audio,
            self._bn_train: False})
    return predicts, scores

  def pred_layers(self, sess, audio):
    dbg_layers, prob, chars, predicts, scores = sess.run(
        [self._dbg_layers, self._prob, self._chars,
         self._pred_cmp, self._score_cmp],
        feed_dict={
            self._audio: audio,
            self._bn_train: False})
    return dbg_layers, prob, chars, predicts, scores

  def pred_embeddings(self, sess, audio):
    dbg_embeddings, chars = sess.run(
        [self._dbg_embeddings, self._chars],
        feed_dict={
            self._audio: audio,
            self._bn_train: False})
    return dbg_embeddings, chars


def eval_bn_params(sess):
  """Evaluates batch norm parameters.
  """
  bn_var = [var for var in tf.global_variables()
            if 'batch_normalization' in var.op.name]
  values = sess.run(bn_var)
  names = [var.op.name for var in bn_var]
  return names, values


def config(parser):
  """Model configurations, shared by training and inference.
  """
  parser.add_argument('--sample_rate', type=int, default=16000)
  parser.add_argument('--frame_size_ms', type=float, default=30.0)
  parser.add_argument('--frame_stride_ms', type=float, default=10.0)
  # fg_interp_factor = target_duration_ms/(target_duration_ms-pad_ms)
  parser.add_argument('--pad_ms', type=float, default=140)
  parser.add_argument('--target_duration_ms', type=int, default=1140)
  parser.add_argument('--num_mel_bins', type=int, default=46)

  parser.add_argument('--file_words', type=str, default='map_words.txt')
  parser.add_argument('--file_chars', type=str, default='map_chars.txt')
  # the top num_key_words in map_words.txt are key words
  parser.add_argument('--num_key_words', type=int, default=10)

  parser.add_argument('--lr_init', type=float, default=0.0002)
  parser.add_argument('--lr_drate', type=float, default=0.3)
  parser.add_argument('--dropout_keep_prob', type=float, default=0.5)
  parser.add_argument('--l2_scale', type=float, default=0)
