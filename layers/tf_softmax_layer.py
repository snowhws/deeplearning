#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from .base_tf_layer import BaseTFLayer


class TFSoftmaxLayer(BaseTFLayer):
    '''多分类层的封装
    支持加一层dropout，大部分模型最后会带一层dropout
    '''
    def __init__(self,
                 mode,
                 in_hidden):
        '''初始化

        Args:
            mode: 训练(train) or 预测(infer)模式，训练模式会返回loss
            in_hidden: 输入层
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            emb_size: 词向量维数
        '''
        self.in_hidden = in_hidden
        self.hidden_size = in_hidden.get_shape()[-1]

    def layer(self):
        '''softmax层
        '''
        with tf.name_scope("fc_output_layer"):
            W = tf.get_variable(
                "W",
                shape=[self.in_hidden.get_shape()[1], self.cls_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b',
                                shape=[self.cls_num],
                                initializer=tf.constant_initializer(0.1))
            if self.mode == "train":
                self.l2_loss += tf.nn.l2_loss(W)
                self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(input_layer, W, b, name="logits")
            self.predictions = tf.nn.softmax(self.logits)
            self.results = self.get_results()

        # loss
        if self.mode == "train":
            self.loss = self.cal_loss() + self.l2_reg_lambda * self.l2_loss
