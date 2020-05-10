#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../../")
import tensorflow as tf
from tf_base_classifier import TFBaseClassifier
from layers.tf_embedding_layer import TFEmbeddingLayer
from layers.tf_bilstm_att_layer import TFBILSTMAttLayer
from layers.tf_classifier_layer import TFClassifierLayer


class TFBILSTMATTClassifier(TFBaseClassifier):
    '''BILSTM-ATTENTION分类器
    '''
    def __init__(self,
                 config,
                 hidden_sizes,
                 attention_size,
                 vocab_size=None,
                 pretrain_word_vecs=None):
        '''
        Args:
            config: 模型配置参数（词典类型），包含learning_rate、classifier_type等参数
            hidden_sizes: 多层BILSTM中每层隐层维数大小
            attention_size: 注意力矩阵宽度
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            pretrain_word_vecs：预训练词向量
        '''
        # 初始化基类
        TFBaseClassifier.__init__(self, config, vocab_size, pretrain_word_vecs)
        # 此分类器参数
        self.hidden_sizes = hidden_sizes
        self.attention_size = attention_size

    def build_model(self):
        '''构建模型
        '''
        embedding_layer = TFEmbeddingLayer(self.input_x, self.vocab_size,
                                           self.emb_size,
                                           self.pretrain_word_vecs).build()
        bilstmatt_layer = TFBILSTMAttLayer(embedding_layer, self.hidden_sizes,
                                           self.attention_size,
                                           self.keep_prob).build()
        self.predictions = TFClassifierLayer(self.mode, bilstmatt_layer,
                                             self.cls_num, self.cls_type,
                                             self.input_y).build()
