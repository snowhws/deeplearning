#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from tf_base_layer import TFBaseLayer
from tf_ln_layer import TFLNLayer


class TFFeedForwardLayer(TFBaseLayer):
    '''Transformer中前馈神经网络层封装
    两层全连接，再加上ResNet。
    '''
    def __init__(self,
                 in_hidden,
                 num_units=[2048, 512],
                 scope="positionwise_feedforward"):
        '''position-wise feed forward net.

        Args:
            in_hidden: 输入层 A 3d tensor with shape of [N, T, C].
            num_units: 两层全连接的隐层大小A list of two integers.
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前层参数
        self.in_hidden = in_hidden
        self.num_units = num_units
        self.scope = scope

    def build(self):
        '''前馈神经网络
        '''
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # Inner: 第一层全连接relu激活
            outputs = tf.layers.dense(self.in_hidden,
                                      self.num_units[0],
                                      activation=tf.nn.relu)

            # Outer: 第二层线性变换
            outputs = tf.layers.dense(outputs, self.num_units[1])

            # Residual connection
            outputs += self.in_hidden

            # Normalize
            outputs = TFLNLayer(outputs).build()

            return outputs
