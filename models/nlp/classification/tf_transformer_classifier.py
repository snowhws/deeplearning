#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../../")
import tensorflow as tf
from tf_base_classifier import TFBaseClassifier
from layers.tf_embedding_layer import TFEmbeddingLayer
from layers.tf_pos_encoding_layer import TFPosEncodingLayer
from layers.tf_multihead_att_layer import TFMultiHeadAttLayer
from layers.tf_feedforward_layer import TFFeedForwardLayer
from layers.tf_classifier_layer import TFClassifierLayer


class TFTransformerClassifier(TFBaseClassifier):
    '''Multi-Head Attention分类器
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
        self.dropout_rate = 1.0 - self.flags.keep_prob
        if not self.flags.training:
            self.dropout_rate = 0.

    def build_model(self):
        '''构建模型
        '''
        embedding_layer = TFEmbeddingLayer(self.input_x, self.flags.vocab_size,
                                           self.flags.emb_size,
                                           self.pretrain_word_vecs).build()

        # add pos encoding
        embedding_layer += TFPosEncodingLayer(embedding_layer,
                                              self.flags.max_seq_len).build()
        embedding_layer = tf.layers.dropout(embedding_layer,
                                            self.dropout_rate,
                                            training=self.flags.training)
        # Transformer Blocks
        encoder = embedding_layer
        for i in range(self.flags.num_blocks):
            with tf.variable_scope("num_blocks_{}".format(i),
                                   reuse=tf.AUTO_REUSE):
                # multihead attention
                encoder = TFMultiHeadAttLayer(queries=encoder,
                                              keys=encoder,
                                              dropout_rate=self.dropout_rate,
                                              training=self.flags.training,
                                              causality=False).build()

                # FFN
                encoder = TFFeedForwardLayer(
                    encoder,
                    [self.flags.hidden_size, self.flags.emb_size]).build()

        # reshape
        concat_layer = tf.reshape(
            encoder, [-1, self.flags.max_seq_len * self.flags.emb_size])

        # loss
        self.probability, self.logits, self.loss = TFClassifierLayer(
            self.flags.training, concat_layer, self.flags.cls_num,
            self.flags.cls_type, self.input_y, self.flags.keep_prob,
            self.flags.l2_reg_lambda).build()

        # 返回模型引用
        return self
