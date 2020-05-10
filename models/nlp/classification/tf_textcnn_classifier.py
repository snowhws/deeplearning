#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

from ....utils.tf_utils import Utils
import tensorflow as tf
from .tf_base_classifier import TFBaseClassifier
from ....layers.tf_embedding_layer import TFEmbeddingLayer
from ....layers.tf_textcnn_layer import TFTextCNNLayer
from ....layers.tf_classifier_layer import TFClassifierLayer


class TFTextCNNClassifier(TFBaseClassifier):
    '''TextCNN分类器
    '''
    def __init__(self, config, filter_sizes, num_filters, vocab_size=None, 
                 pretrain_word_vecs=None):
        '''
        Args:
            config: 模型配置参数（词典类型），包含learning_rate、classifier_type等参数
            max_seq_len: 序列最大长度
            filter_sizes: array类型，所有卷积核的大小，支持多个窗口同时卷积
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            pretrain_word_vecs：预训练词向量
        '''
        # 初始化基类
        TFBaseClassifier.__init__(self, config, vocab_size, pretrain_word_vecs)    
        # 此分类器参数
        self.filter_sizes = filter_sizes
        self.max_seq_len = max_seq_len

    def build_model(self):
        '''构建模型
        '''
        embedding_layer = TFEmbeddingLayer(self.input_x, self.vocab_size, 
                                           self.emb_size, self.pretrain_word_vecs)
        textcnn_layer = TFTextCNNLayer(embedding_layer, self.max_seq_len, 
                                       self.filter_sizes, self.num_filters)
        self.predictions = TFClassifierLayer(self.mode, textcnn_layer, 
                                             self.cls_num, self.cls_type, self.input_y)
        
