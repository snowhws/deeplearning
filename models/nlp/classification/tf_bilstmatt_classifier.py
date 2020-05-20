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
        embedding_layer = TFEmbeddingLayer(self.input_x, self.flags.vocab_size,
                                           self.flags.emb_size,
                                           self.flags.keep_prob,
                                           self.flags.training,
                                           self.pretrain_word_vecs).build()
        # [B, T, D] -> [B, H]
        bilstmatt_layer = TFBILSTMAttLayer(embedding_layer, self.hidden_sizes,
                                           self.flags.attention_size,
                                           self.flags.keep_prob,
                                           self.flags.training).build()
        self.probability, self.logits, self.loss = TFClassifierLayer(
            self.flags.training, bilstmatt_layer, self.flags.cls_num,
            self.flags.cls_type, self.input_y, self.flags.keep_prob,
            self.flags.l2_reg_lambda).build()

        return self
