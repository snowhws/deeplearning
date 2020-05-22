#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import tensorflow as tf
from tf_base_classifier import TFBaseClassifier
from layers.tf_embedding_layer import TFEmbeddingLayer
from layers.tf_bilstm_att_layer import TFBILSTMAttLayer
from layers.tf_classifier_layer import TFClassifierLayer


class TFBILSTMATTClassifier(TFBaseClassifier):
    '''BILSTM-ATTENTION分类器
    '''
    def __init__(self, flags):
        '''
        Args:
            flags: 全局参数，包含learning_rate、classifier_type等参数
        '''
        # 初始化基类
        TFBaseClassifier.__init__(self, flags)
        # 此分类器参数
        self.hidden_sizes = list(map(int, flags.hidden_sizes.split(",")))

    def build_model(self):
        '''构建模型
        '''
        # [B, T_w] -> [B, T, D]
        embedding_layer = TFEmbeddingLayer(
            input_x=self.input_x,
            vocab_size=self.flags.vocab_size,
            emb_size=self.flags.emb_size,
            keep_prob=self.flags.keep_prob,
            training=self.flags.training,
            pretrain_word_vecs=self.pretrain_word_vecs).build()
        # [B, T, D] -> [B, H]
        bilstmatt_layer = TFBILSTMAttLayer(
            in_hidden=embedding_layer,
            hidden_sizes=self.hidden_sizes,
            attention_size=self.flags.attention_size,
            keep_prob=self.flags.keep_prob,
            training=self.flags.training,
            rnn_type=self.flags.rnn_type).build()
        # [B, H] -> [B, cls_num]
        self.probability, self.logits, self.loss = TFClassifierLayer(
            training=self.flags.training,
            in_hidden=bilstmatt_layer,
            cls_num=self.flags.cls_num,
            cls_type=self.flags.cls_type,
            input_y=self.input_y,
            keep_prob=self.flags.keep_prob,
            l2_reg_lambda=self.flags.l2_reg_lambda).build()

        return self
