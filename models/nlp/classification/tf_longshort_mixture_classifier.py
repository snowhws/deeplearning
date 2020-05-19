#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import tensorflow as tf
from tf_base_classifier import TFBaseClassifier
from layers.tf_embedding_layer import TFEmbeddingLayer
from layers.tf_hierarchical_att_layer import TFHierarchicalAttLayer
from layers.tf_bilstm_att_layer import TFBILSTMAttLayer
from layers.tf_classifier_layer import TFClassifierLayer


class TFLongShortMixtureClassifier(TFBaseClassifier):
    '''长短文本混合建模分类器：适合新闻title+content联合建模。
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
        # title短文本建模
        # [B, T_w] -> [B, T_w, D]
        title_embedding = TFEmbeddingLayer(self.input_x, self.flags.vocab_size,
                                           self.flags.emb_size,
                                           self.pretrain_word_vecs).build()
        # [B, T_w, D] -> [B, last_H]
        bilstmatt_layer = TFBILSTMAttLayer(title_embedding, self.hidden_sizes,
                                           self.flags.attention_size,
                                           self.flags.keep_prob,
                                           self.flags.training).build()
        # content长文本建模
        # [B, T_s, T_w] -> [B, T_s, T_w, D]
        content_embedding = TFEmbeddingLayer(self.input_c,
                                             self.flags.vocab_size,
                                             self.flags.emb_size,
                                             self.pretrain_word_vecs).build()
        # [B, T_s, T_w, D] -> [B, H]
        hierarchical_layer = TFHierarchicalAttLayer(
            content_embedding, self.flags.max_doc_len, self.flags.max_seq_len,
            self.flags.hidden_size, self.flags.attention_size,
            self.flags.keep_prob, self.flags.training).build()

        # concat
        mix_layer = tf.concat([bilstmatt_layer, hierarchical_layer], axis=1)

        # -> [cls_num]
        self.probability, self.logits, self.loss = TFClassifierLayer(
            self.flags.training, mix_layer, self.flags.cls_num,
            self.flags.cls_type, self.input_y, self.flags.keep_prob,
            self.flags.l2_reg_lambda).build()

        return self