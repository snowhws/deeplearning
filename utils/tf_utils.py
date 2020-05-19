#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author wensong

import tensorflow as tf
import numpy as np
from tqdm import tqdm
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
    def tokenizer_fn(strs):
        '''按空格切分，语料需要事先处理好
        '''
        for line in strs:
            yield line.split()

    @staticmethod
    def one_hot_emb(idx, cls_num):
        '''生成one-hot表示的数组
        '''
        one_hot = []
        for i in range(cls_num):
            if i == idx:
                one_hot.append(1)
            else:
                one_hot.append(0)
        return one_hot

    @staticmethod
    def get_sequence_lens(sequences):
        '''获取一个batch中序列的真实长度，提供给bilstm加速

        Args:
            sequences: [B, T, D]

        Returns:
            [B]
        '''
        # [B, T, D] -> [B, T]
        used = tf.sign(tf.reduce_max(tf.abs(sequences), axis=2))
        # [B, T] -> [B]
        lens = tf.reduce_sum(used, axis=1)

        return tf.cast(lens, tf.int32)

    @staticmethod
    def preprocess(strs, vocab_set, seg_sent=False, encoding="utf-8"):
        '''字符串预处理
        '''
        seg_dict = set([".", ";", "?", "？", "!", "。", "；", "！"])
        text = strs.encode(encoding, "ignore").strip().rstrip().lower()
        words = text.split()
        vocab_set |= set(words)
        if seg_sent:
            sents = []
            sent = ""
            for word in words:
                sent += word + " "
                if word in seg_dict:
                    sents.append(sent.rstrip())
                    sent = ""
            if len(sent) > 2:
                sents.append(sent.rstrip())
            return sents
        return text

    @staticmethod
    def load_shorttext_data(shorttext_file, encoding='utf-8'):
        '''加载正负样本，样本每行为分词后的句子。

        Returns:
            输出vocab_set, texts, labels
        '''
        # file => array
        examples = list(
            io.open(shorttext_file, "r", encoding=encoding).readlines())
        # preprocess
        vocab_set = set()
        samples = [TFUtils.preprocess(s, vocab_set) for s in examples]
        # labels, sents
        labels = []
        sents = []
        for sample in samples:
            label, sent = sample.split("\t")
            labels.append(TFUtils.one_hot_emb(int(label), cls_num))
            sents.append(sent)
        return vocab_set, sents, labels

    @staticmethod
    def load_longtext_with_title_data(longtext_file,
                                      cls_num,
                                      encoding='utf-8'):
        '''输入[1(class), title, content]

        Returns:
            输出vocab_set, titles, contents, labels
            contents is an array [sent, ...]
        '''
        # file => array
        samples = list(
            io.open(longtext_file, 'r', encoding=encoding).readlines())
        # labels, titles, contents
        vocab_set = set()
        labels = []
        titles = []
        contents = []
        for sample in samples:
            # split sample
            label, title, content = sample.split("\t")
            # preprocess
            title = TFUtils.preprocess(title, vocab_set)
            sents = TFUtils.preprocess(content, vocab_set, seg_sent=True)
            # add
            labels.append(TFUtils.one_hot_emb(int(label), cls_num))
            titles.append(title)
            contents.append(sents)
        return vocab_set, titles, contents, labels

    @staticmethod
    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        '''Generates a batch iterator for a dataset.
        '''
        # 所有数据
        if len(data) == 2:
            data = np.array(list(zip(data[0], data[1])))
        elif len(data) == 3:
            data = np.array(list(zip(data[0], data[1], data[2])))
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

    @staticmethod
    def cut_and_padding_2D(matrix, row_lens, col_lens):
        '''补齐二维数组
        '''
        # cut截断
        if len(matrix) >= row_lens:
            return matrix[:row_lens]
        # padding
        row = np.array([0 for i in range(col_lens)])
        count = row_lens - len(matrix)
        for i in range(count):
            matrix = np.insert(matrix, -1, values=row, axis=0)
        return matrix
