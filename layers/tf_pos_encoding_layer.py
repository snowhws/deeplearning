#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
import numpy as np
from tf_base_layer import TFBaseLayer


class TFPosEncodingLayer(TFBaseLayer):
    '''Transformer中使用正弦对位置进行编码
    Positional Encoding Layer
    '''
    def __init__(self,
                 in_hidden,
                 max_seq_len,
                 masking=True,
                 scope="positional_encoding"):
        '''positional encoding

        Args:
            in_hidden: 输入层 A 3d tensor with shape of [N, T, C].
            max_seq_len: 最大长度
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前层参数
        self.in_hidden = in_hidden
        self.max_seq_len = max_seq_len
        self.masking = masking
        self.scope = scope

    def build(self):
        '''位置编码
        '''
        E = self.in_hidden.get_shape().as_list()[-1]  # static
        N, T = tf.shape(self.in_hidden)[0], tf.shape(
            self.in_hidden)[1]  # dynamic
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0),
                                   [N, 1])  # (N, T)

            # First part of the PE function: sin and cos argument
            position_enc = np.array(
                [[pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                 for pos in range(self.max_seq_len)])

            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc,
                                                tf.float32)  # (maxlen, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if self.masking:
                outputs = tf.where(tf.equal(self.in_hidden, 0), self.in_hidden,
                                   outputs)

            return outputs
