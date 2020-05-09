#!/usr/bin/env python
# coding: utf-8
# @author: wensong

import tensorflow as tf
from .base_tf_layer import BaseTFLayer


class TFTextCNNLayer(BaseTFLayer):
    '''TextCNN Layer
    底层embedding layer, 再接多窗口多核卷积，最后最大池化max-pooling
    '''
    def __init__(self, in_hidden, max_seq_len, filter_sizes, num_filters):
        '''TextCNN初始化

        Args:
            in_hidden: 输入层tensor, 通常是一个batch的词向量
            max_seq_len: 序列最大长度
            filter_sizes: array类型，所有卷积核的大小，支持多个窗口同时卷积
            num_filters: 卷积核个数
        '''
        # 父类初始化
        BaseTFLayer.__init__(self)
        # 参数
        self.in_hidden = self.in_hidden
        self.emb_size = self.in_hidden.get_shape()[-1]
        self.max_seq_len = max_seq_len
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    def layer(self):
        '''TextCNN Layer层

        Returns:
            返回经过TextCNN后的隐层表示，shape是[batch, feature_dim=filter_sizes*num_filters]
        '''
        # 在-1列扩展一维，tf.nn.conv2d的input参数为四维变量
        # shape: [batch_size, seq_len, emb_size, 1]
        embedded_words_expanded = tf.expand_dims(self.in_hidden, -1)

        # 所有卷积核的池化层
        pooled_outputs = []
        # 遍历卷积核：可以同时用3、4、5等多个窗口
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积核shape：[卷积核大小，宽度，输入通数，输出通道数]
                filter_shape = [
                    filter_size, self.emb_size, 1, self.num_filters
                ]
                # 卷积核
                # 随机生成截断正态分布参数，大于两倍标准差stddev即截断
                filters = tf.Variable(tf.truncated_normal(filter_shape,
                                                          stddev=0.1),
                                      name="filters")
                W = tf.get_variable(
                    "W" + str(filter_size),
                    shape=filter_shape,
                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b" + str(filter_size),
                                    shape=[self.num_filters],
                                    initializer=tf.zeros_initializer())
                # 卷积
                # 'SAME'为等长卷积填充0, 'VALID'为窄卷积不填充
                conv = tf.nn.conv2d(embedded_words_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv" + str(filter_size))
                # 非线性变换隐层
                hidden = tf.nn.relu(tf.add(conv, b), name="relu")
                # 最大池化
                pooled = tf.nn.max_pool(
                    hidden,
                    ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool" + str(filter_size))
                # 池化后的结果append起来
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        feature_dim = self.num_filters * len(self.filter_sizes)
        # 在out_channel维度拼接
        # [batch, emb_size, in_channel_num, out_channel_num]
        h_pool = tf.concat(pooled_outputs, 3)
        # reshape: [batch, feature_dim]
        self.output = tf.reshape(h_pool, [-1, feature_dim])

        return self.output
