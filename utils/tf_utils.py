#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author wensong

import tensorflow as tf
import numpy as np
import re
import sys
import io
import logging


class TFUtils(object):
    '''工具类
    '''
    @staticmethod
    def default_dict(dct, key, dft_value):
        '''根据key返回dict内的value，否则返回默认value
        '''
        if key not in dct:
            return dft_value

        return dct[key]

    @staticmethod
    def nonzero_indices(inputs):
        '''获取张量非零索引

        Returns:
            返回索引张量
        '''
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(inputs, zero)
        indices = tf.where(where)

        return indices

    @staticmethod
    def preprocess(strs):
        '''字符串预处理
        '''
        strs = strs.strip().rstrip().lower()
        return strs

    @staticmethod
    def load_data_and_labels(positive_data_file, negative_data_file):
        '''加载样本、分词、打label

        Returns:
            words and labels.
        '''
        # Load data from files
        # file => array
        positive_examples = list(
            io.open(positive_data_file, "r", encoding='utf-8').readlines())
        positive_examples = [TFUtils.preprocess(s) for s in positive_examples]
        negative_examples = list(
            io.open(negative_data_file, "r", encoding='utf-8').readlines())
        negative_examples = [TFUtils.preprocess(s) for s in negative_examples]
        # Split by words
        texts = positive_examples + negative_examples
        # Generate labels
        positive_labels = [[0, 1] for _ in positive_examples]
        negative_labels = [[1, 0] for _ in negative_examples]
        # 按列拼接
        labels = np.concatenate([positive_labels, negative_labels], 0)
        # format is: [text, labels]
        return [texts, labels]

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        '''Generates a batch iterator for a dataset.
        '''
        # 所有数据
        data = np.array(data)
        data_size = len(data)
        # batch个数计算
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        # 每个epoch
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                np.random.shuffle(data)
            # 遍历batch
            for batch_index in range(num_batches_per_epoch):
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, data_size)
                # 使用yield动态返回batch
                yield data[start_index:end_index]
