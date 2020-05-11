#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../../")
from utils.tf_utils import TFUtils
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import logging


class PreProcessor(object):
    '''nlp分类器初始化
    '''
    def __init__(self):
        '''init
        '''
        self.name = "PRE"

    def execute(self, params):
        '''预处理数据
        '''
        # 脚本参数
        flags = params["INIT"]

        # 加载样本
        x_text, y = TFUtils.load_data_and_labels(flags.positive_data_file,
                                                 flags.negative_data_file)

        # 构建词表
        max_document_length = flags.max_seq_len
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length)
        x = np.array(list(vocab_processor.fit_transform(x_text)))
        flags.vocab_size = len(vocab_processor.vocabulary_) + 1

        # 随机打乱数据
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = x[shuffle_indices]
        y_shuffled = y[shuffle_indices]

        # 分割出训练和测试集合
        dev_sample_index = -1 * int(
            flags.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[
            dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[
            dev_sample_index:]

        logging.info("Vocabulary Size: {:d}".format(
            len(vocab_processor.vocabulary_)))
        logging.info("Train/Dev split: {:d}/{:d}".format(
            len(y_train), len(y_dev)))

        return x_train, y_train, vocab_processor, x_dev, y_dev
