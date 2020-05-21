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

        # 不同分类任务，选择不同预处理方式
        ret = self._pre_text_to_ids(flags)

        # 打印参数
        self._print_flags(flags)

        return ret

    def _print_flags(self, flags):
        '''打印所有参数
        '''
        for key in flags.flag_values_dict():
            logging.info("FLAGS " + key + " : " + str(flags[key].value))

    def _pre_text_to_ids(self, flags):
        # [titles, contents, labels]
        vocab_set, titles, conts, labels = TFUtils.load_multitype_text(
            flags.data_type, flags.data_file, flags.cls_num,
            flags.doc_separators)
        # 词表处理器
        vocab_processor = learn.preprocessing.VocabularyProcessor(
            max_document_length=flags.max_seq_len,
            tokenizer_fn=TFUtils.tokenizer_fn)
        # 建立词表
        vocab_processor.fit(list(vocab_set))
        # 转化索引
        tids = None
        if flags.data_type == "shorttext" or flags.data_type == "longtext_with_title":
            tids = np.array(list(vocab_processor.transform(titles)))
        cids = []
        if flags.data_type == "longtext" or flags.data_type == "longtext_with_title":
            for sents in conts:
                sids = np.array(list(vocab_processor.transform(sents)))
                # cut & padding长文本
                sids = TFUtils.cut_and_padding_2D(matrix=sids,
                                                  row_lens=flags.max_doc_len,
                                                  col_lens=flags.max_seq_len)
                sids = np.array(sids)
                cids.append(sids)
        # 转为np.array类型
        labels = np.array(labels)
        cids = np.array(cids)
        # 词表大小
        flags.vocab_size = len(vocab_processor.vocabulary_) + 1

        # 随机打乱数据
        np.random.seed()
        shuffle_indices = np.random.permutation(np.arange(len(labels)))
        labels_shuffled = labels[shuffle_indices]
        # 分割出训练和测试集合
        dev_sample_index = -1 * int(
            flags.dev_sample_percentage * float(len(labels)))
        l_train, l_dev = labels_shuffled[:dev_sample_index], labels_shuffled[
            dev_sample_index:]
        t_train = None
        t_dev = None
        c_train = None
        c_dev = None
        if flags.data_type == "shorttext" or flags.data_type == "longtext_with_title":
            tids_shuffled = tids[shuffle_indices]
            t_train, t_dev = tids_shuffled[:dev_sample_index], tids_shuffled[
                dev_sample_index:]
        if flags.data_type == "longtext" or flags.data_type == "longtext_with_title":
            cids_shuffled = cids[shuffle_indices]
            c_train, c_dev = cids_shuffled[:dev_sample_index], cids_shuffled[
                dev_sample_index:]

        logging.info("Train/Dev split: {:d}/{:d}".format(
            len(l_train), len(l_dev)))

        # 组成tuple
        train_tuple = (t_train, c_train, l_train)
        test_tuple = (t_dev, c_dev, l_dev)

        return train_tuple, vocab_processor, test_tuple
