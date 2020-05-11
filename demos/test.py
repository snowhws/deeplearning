#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_string('data_dir', '/tmp/mnist', 'Directory with the MNIST data.')
flags.DEFINE_integer('batch_size', 5, 'Batch size.')
flags.DEFINE_integer('num_evals', 1000, 'Number of batches to evaluate.')
FLAGS = flags.FLAGS

print(FLAGS.data_dir, FLAGS.batch_size, FLAGS.num_evals)
