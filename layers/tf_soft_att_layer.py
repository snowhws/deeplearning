#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from tf_base_layer import TFBaseLayer


class TFSoftAttLayer(TFBaseLayer):
    '''soft attention层封装
    参考paper(Hierarchical Attention Networks for Document Classification)：https://www.aclweb.org/anthology/N16-1174/
    softmax求出attention score后，对隐层进行软加权。
    '''
    def __init__(self,
                 in_hidden,
                 attention_size,
                 training=True,
                 scope="soft_attention"):
        '''初始化

        Args:
            in_hidden: 需要进行软加权的隐层
            attention_size: attention权重矩阵宽度
            training: 是否训练模式
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前层参数
        self.in_hidden = in_hidden
        self.in_hidden_size = in_hidden.get_shape()[-1]
        self.attention_size = attention_size
        self.training = training
        self.scope = scope

    def build(self):
        """返回soft-attention后的向量表示
        输入Shape为[Batch, TimeStep, In_Hidden_Size]

        Returns:
            返回shape为[Batch, In_Hidden_Size]
        """
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # 初始化att参数
            att_w = tf.get_variable(
                "attention_weight",
                shape=[self.in_hidden_size, self.attention_size],
                initializer=tf.contrib.layers.xavier_initializer())
            att_b = tf.get_variable("attention_bias",
                                    shape=[self.attention_size],
                                    initializer=tf.zeros_initializer())
            att_u = tf.get_variable(
                "attention_u",
                shape=[self.attention_size],
                initializer=tf.contrib.layers.xavier_initializer())

            # 非线性转换
            # [B, T, H] dot [H, A] = [B, T, A]
            att_v = tf.tanh(
                tf.tensordot(self.in_hidden, att_w, axes=1) + att_b)

            # [B, T, A] dot [A] = [B, T]
            att_vu = tf.tensordot(att_v, att_u, axes=1, name='attention_vu')

            # attention score, [B, T]
            att_alpha = tf.nn.softmax(att_vu, name='attention_alpha')

            # 如果是训练过程，则添加attention图像summary
            if self.training:
                # 取batch中一个样例：[N, T] -> [1, T]
                attention = att_alpha[:1]
                # [batch, height, width, channels]
                # channels = 1: 灰度
                # channels = 3: RGB
                # channels = 4: RGBA
                # 扩展channels: [1, 1, T, 4]
                attention = tf.expand_dims(tf.expand_dims(attention, 1), -1)
                attention = tf.tile(attention, [1, 1, 1, 4])
                # tensorboard images上显示attention效果图
                tf.summary.image("attention", attention)

            # expand to: [B, T] -> [B, T, 1]
            att_expand = tf.expand_dims(att_alpha, -1)
            # 注意是点乘: [B, T, H] * [B, T, 1] = [B, T, H]
            att_ah = self.in_hidden * att_expand
            # max pooling, axis=1: [B, T, H] -> [B, H]
            output = tf.reduce_max(att_ah, axis=1)

            # [Batch, In_Hidden_Size]
            return output
