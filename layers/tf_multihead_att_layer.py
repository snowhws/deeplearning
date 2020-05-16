#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from tf_base_layer import TFBaseLayer
from tf_ln_layer import TFLNLayer


class TFMultiHeadAttLayer(TFBaseLayer):
    '''Multi-Head Attention层封装
    '''
    def __init__(self,
                 queries,
                 keys,
                 num_heads=8,
                 dropout_rate=0.5,
                 training=True,
                 causality=False,
                 scope="multihead_attention"):
        '''初始化

        Args:
            queries: 查询Q, A 3d tensor with shape of [N, T_q, d_model].
            keys: 匹配K, A 3d tensor with shape of [N, T_k, d_model].
            num_heads: 多头数目
            dropout_rate: dropout丢弃概率 = 1 - keep_prob
            training: 是否在训练，控制dropout层的生效方式
            causality: 是否mask未来预测部分，transformer decoder中使用
            scope: 共享变量variable_scope
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前层参数
        self.queries = queries
        self.keys = keys
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.training = training
        self.causality = causality
        self.scope = scope

    def build(self):
        """构造多头注意力层

        Returns:
            返回经过multi-head attention后的表示
        """
        # attention维度与词向量维度一致，因为后续有res connection
        self.d_model = self.queries.get_shape().as_list()[-1]
        # multihead共享变量
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # QKV做线性变换
            Q = tf.layers.dense(self.queries, self.d_model,
                                use_bias=True)  # (N, T_q, d_model)
            K = tf.layers.dense(self.keys, self.d_model,
                                use_bias=True)  # (N, T_k, d_model)
            V = tf.layers.dense(self.keys, self.d_model,
                                use_bias=True)  # (N, T_k, d_model)

            # 切割和拼接多头
            Q_ = tf.concat(tf.split(Q, self.num_heads, axis=2),
                           axis=0)  # (h*N, T_q, d_model/h)
            K_ = tf.concat(tf.split(K, self.num_heads, axis=2),
                           axis=0)  # (h*N, T_k, d_model/h)
            V_ = tf.concat(tf.split(V, self.num_heads, axis=2),
                           axis=0)  # (h*N, T_k, d_model/h)

            # Attention
            outputs = self._scaled_dot_product_attention(
                Q_,
                K_,
                V_,
                causality=self.causality,
                dropout_rate=self.dropout_rate,
                training=self.training)

            # Restore shape
            outputs = tf.concat(tf.split(outputs, self.num_heads, axis=0),
                                axis=2)  # (N, T_q, d_model)

            # Residual connection
            outputs += self.queries

            # LN归一
            outputs = TFLNLayer(outputs).build()

            return outputs

    def _scaled_dot_product_attention(self,
                                      Q,
                                      K,
                                      V,
                                      causality=False,
                                      dropout_rate=0.,
                                      training=True,
                                      scope="scaled_dot_product_attention"):
        '''softmax之前缩放，并做mask，公式Attention(Q,K,V)=softmax(Q*K^T / √dk)*V
        '''
        with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
            # Q*K^T
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

            # scale: Q*K^T / √dk
            outputs /= self.d_model**0.5

            # 先对key做padding mask
            outputs = self._mask(outputs, name="key")

            # 上三角mask预测词语
            if causality:
                outputs = self._mask(outputs, name="future")

            # softmax(Q*K^T / √dk)
            outputs = tf.nn.softmax(outputs)

            # 如果是训练过程，则添加attention图像summary
            if training:
                # [N, T_q, T_k] -> [N, T_k, T_q]
                attention = tf.transpose(outputs, [0, 2, 1])
                # 取batch中一个样例：[N, T_k, T_q] -> [1, T_k, T_q]
                attention = attention[:1]
                # [batch, height, width, channels]
                # channels = 1: 灰度
                # channels = 3: RGB
                # channels = 4: RGBA
                # 扩展channels: [1, T_k, T_q, 4]
                attention = tf.expand_dims(attention, -1)
                attention = tf.tile(attention, [1, 1, 1, 4])
                # tensorboard images上显示attention效果图
                tf.summary.image("attention", attention)

            # 再对query做mask
            # 这时候不需要给很小值，因为前面outputs已经对key做过，使得softmax后att_score为0
            outputs = self._mask(outputs, name="query")

            # dropout
            outputs = tf.layers.dropout(outputs,
                                        rate=dropout_rate,
                                        training=training)

            # weighted sum (context vectors):  softmax(Q*K^T / √dk) * V
            outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

            return outputs

    def _mask(self, outputs, name):
        """Transformer中两类Mask:
            一种padding mask补齐部分是0值，attention不需要关注，那么求softmax后也应该为0，所以需要mask到负无穷大。
            另一种sequence mask遮盖未来预测部分，对att对齐矩阵上三角进行处理。

        Args:
            inputs: 3d tensor. (h*N, T_q, T_k)
            name: string. "key" | "query" | "future"
        """

        # 负无穷: softmax得出att_score后正好为0
        mask_num = -2**32 + 1

        # key padding mask
        if name == "key":
            key_masks = tf.sign(tf.abs(tf.reduce_sum(self.keys,
                                                     axis=-1)))  # (N, T_k)
            key_masks = tf.tile(key_masks, [self.num_heads, 1])  # (h*N, T_k)
            key_masks = tf.tile(
                tf.expand_dims(key_masks, 1),
                [1, tf.shape(self.queries)[1], 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(outputs) * mask_num  # -infinity
            outputs = tf.where(tf.equal(key_masks, 0), paddings,
                               outputs)  # (h*N, T_q, T_k)
        # query padding mask
        elif name == "query":
            query_masks = tf.sign(tf.abs(tf.reduce_sum(self.queries,
                                                       axis=-1)))  # (N, T_q)
            query_masks = tf.tile(query_masks,
                                  [self.num_heads, 1])  # (h*N, T_q)
            query_masks = tf.tile(
                tf.expand_dims(query_masks, -1),
                [1, 1, tf.shape(self.keys)[1]])  # (h*N, T_q, T_k)
            outputs *= query_masks  # broadcasting. (N, T_q, C)
        elif name == "future":
            # 对att_score对齐矩阵上三角做mask，遮蔽未来词语
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.linalg.LinearOperatorTriL(
                diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0),
                            [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2**32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings,
                               outputs)  # (h*N, T_q, T_k)

        return outputs
