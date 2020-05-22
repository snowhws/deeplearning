#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import logging
import tensorflow as tf
from tf_base_classifier import TFBaseClassifier
from layers.tf_embedding_layer import TFEmbeddingLayer
from layers.tf_textcnn_layer import TFTextCNNLayer
from layers.tf_classifier_layer import TFClassifierLayer


class TFTextCNNClassifier(TFBaseClassifier):
    '''TextCNN分类器
    '''
    def __init__(self, flags):
        '''
        Args:
            flags: 全局参数，包含learning_rate、classifier_type等参数
        '''
        # 初始化基类
        TFBaseClassifier.__init__(self, flags)
        # 此分类器参数
        self.filter_sizes = list(map(int, flags.filter_sizes.split(",")))

    def build_model(self):
        '''构建模型
        '''
        # [B, T] -> [B, T, D]
        embedding_layer = TFEmbeddingLayer(
            input_x=self.input_x,
            vocab_size=self.flags.vocab_size,
            emb_size=self.flags.emb_size,
            keep_prob=self.flags.keep_prob,
            training=self.flags.training,
            pretrain_word_vecs=self.pretrain_word_vecs).build()
        # [B, T, D] -> [B, H]
        textcnn_layer = TFTextCNNLayer(in_hidden=embedding_layer,
                                       max_seq_len=self.flags.max_seq_len,
                                       filter_sizes=self.filter_sizes,
                                       num_filter=self.flags.num_filters,
                                       training=self.flags.training).build()
        # [B, H] -> [B, cls_num]
        self.probability, self.logits, self.loss = TFClassifierLayer(
            training=self.flags.training,
            in_hidden=textcnn_layer,
            cls_num=self.flags.cls_num,
            cls_type=self.flags.cls_type,
            input_y=self.input_y,
            keep_prob=self.flags.keep_prob,
            l2_reg_lambda=self.flags.l2_reg_lambda).build()
        return self
