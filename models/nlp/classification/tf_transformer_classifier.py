#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
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
        embedding_layer = TFEmbeddingLayer(
            input_x=self.input_x,
            vocab_size=self.flags.vocab_size,
            emb_size=self.flags.emb_size,
            keep_prob=self.flags.keep_prob,
            training=self.flags.training,
            pretrain_word_vecs=self.pretrain_word_vecs).build()

        # add pos encoding
        embedding_layer += TFPosEncodingLayer(
            in_hidden=embedding_layer,
            max_seq_len=self.flags.max_seq_len).build()
        embedding_layer = tf.layers.dropout(embedding_layer,
                                            dropout_rate=self.dropout_rate,
                                            training=self.flags.training)
        # Transformer Blocks
        encoder = embedding_layer  # [B, T, D]
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
                    in_hidden=encoder,
                    num_units=[self.flags.hidden_size,
                               self.flags.emb_size]).build()

        # mean or max pooling: [B, T, D] -> [B, D]
        encoder = tf.reduce_max(encoder, axis=1)

        # [B, D] -> [B, cls_num]
        self.probability, self.logits, self.loss = TFClassifierLayer(
            training=self.flags.training,
            in_hidden=encoder,
            cls_num=self.flags.cls_num,
            cls_type=self.flags.cls_type,
            input_y=self.input_y,
            keep_prob=self.flags.keep_prob,
            l2_reg_lambda=self.flags.l2_reg_lambda).build()

        # 返回模型引用
        return self
