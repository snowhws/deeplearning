#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from tf_base_layer import TFBaseLayer


class TFLNLayer(TFBaseLayer):
    '''Layer Normalization归一化层封装
    区别于batch noramlization，LN按每一条样本的均值和标准差做归一化
    '''
    def __init__(self, in_hidden, epsilon=1e-8, scope="layer_normalization"):
        '''初始化

        Args:
            in_hidden: 输入层
            epsilon: 公式中分母加和部分
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前层参数
        self.in_hidden = in_hidden
        self.epsilon = epsilon
        self.scope = scope

    def build(self):
        """区别于batch noramlization，LN按每一条样本的均值和标准差做归一化

        Returns:
            归一化后的向量表示
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            inputs_shape = self.in_hidden.get_shape()
            params_shape = inputs_shape[-1:]

            # 均值
            mean, variance = tf.nn.moments(self.in_hidden, [-1],
                                           keep_dims=True)
            beta = tf.get_variable("beta",
                                   params_shape,
                                   initializer=tf.zeros_initializer())
            gamma = tf.get_variable("gamma",
                                    params_shape,
                                    initializer=tf.ones_initializer())
            # gamma * (input - mean)/sqrt(std + epsilon) + beta
            normalized = (self.in_hidden - mean) / (
                (variance + self.epsilon)**.5)
            output = gamma * normalized + beta

            return output
