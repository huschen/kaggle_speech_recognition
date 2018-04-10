"""The main inference script, to load the trained model and run predictions
  on the test set.
"""

import argparse
import sys
import glob
import os

import tensorflow as tf

import util_data
import util_train
import models

FLAGS = None


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  word_dict = util_data.WordDict(FLAGS.file_words, FLAGS.file_chars,
                                 FLAGS.num_key_words)
  _, num_char_classes = word_dict.num_classes
  max_label_length = word_dict.max_label_length
  audio_length = int(FLAGS.sample_rate * FLAGS.target_duration_ms / 1000)

  model_graph = models.AudioInferer(FLAGS, num_char_classes, max_label_length)
  saver = tf.train.Saver(tf.global_variables())

  sess = util_train.tf_session(gmem_dynamic=True)
  tf.train.write_graph(sess.graph_def, FLAGS.result_dir,
                       'audio_inference_graph.pbtxt')

  # training +validation data
  wav_list = glob.glob(os.path.join(FLAGS.train_data_dir,
                                    '*', '*nohash*.wav'))
  util_train.infer_testset(model_graph, sess, audio_length, word_dict,
                           wav_list, FLAGS.result_dir, FLAGS.batch_size,
                           saver, FLAGS.ckpt, mode='train')
  # test data
  wav_list = glob.glob(os.path.join(FLAGS.test_data_dir, '*wav'))
  util_train.infer_testset(model_graph, sess, audio_length, word_dict,
                           wav_list, FLAGS.result_dir, FLAGS.batch_size,
                           saver, FLAGS.ckpt)
  sess.close()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--result_dir', type=str, default='run')
  parser.add_argument('--ckpt', type=str,
                      default='run/ckpts/ann_0_0.ckpt-0')
  parser.add_argument('--test_data_dir', type=str,
                      default='dataset/test/audio/')
  parser.add_argument('--train_data_dir', type=str,
                      default='dataset/train/audio/')
  parser.add_argument('--batch_size', type=int, default=128)

  models.config(parser)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
