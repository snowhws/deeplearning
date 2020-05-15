#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import logging
sys.path.append(os.getcwd() + "/../../")
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
        embedding_layer = TFEmbeddingLayer(self.input_x, self.flags.vocab_size,
                                           self.flags.emb_size,
                                           self.pretrain_word_vecs).build()
        textcnn_layer = TFTextCNNLayer(embedding_layer, self.flags.max_seq_len,
                                       self.filter_sizes,
                                       self.flags.num_filters).build()
        self.probability, self.logits, self.loss = TFClassifierLayer(
            self.flags.training, textcnn_layer, self.flags.cls_num,
            self.flags.cls_type, self.input_y, self.flags.keep_prob,
            self.flags.l2_reg_lambda).build()
        return self
