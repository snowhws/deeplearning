#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np


# CNN分类器封装
class TextCNN(object):
    '''
    CNN Classifier
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    '''
    def __init__(self, seq_len, cls_num, voc_size, emb_size, filter_sizes,
                 num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, seq_len], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, cls_num], name="input_y")
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of l2 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer
        # 定义命名空间，方便可视化展示：tf.name_scope("embedding")
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # 整个emb空间
            # 产生一个voc_size*emb_size的矩阵，范围是[-1.0, 1.0)
            # tf.random_uniform([voc_size, emb_size], -1.0, 1.0)
            self.emb = tf.Variable(tf.random_uniform([voc_size, emb_size], -1.0, 1.0), name="emb")
            # 根据索引查询张量
            self.embedded_chars = tf.nn.embedding_lookup(self.emb, self.input_x)
            # 在-1列最后一个位置添加一维，值为1
            # 此处整个shape是 [句子数量batch, seq_len, emb_size, 1]，目的是方便卷积操作，因为tf.nn.conv2d的input参数为四维变量
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = [] # 所有卷积输出池
        # 第i个过滤器size大小
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                # 卷积核shape：[卷积核的高度，词向量维度（卷积核的宽度），1（输入通道数），卷积核个数（输出通道数）]
                filter_shape = [filter_size, emb_size, 1, num_filters]
                # 卷积核
                # tf.truncated_normal(filter_shape, stddev=0.1): 随机生成截断正态分布参数，大于两倍标准差stddev即截断
                filters = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filters")
                # 偏移量：每个卷积核一个bias
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                # 卷积
                conv = tf.nn.conv2d(
                    # 输入：  [句子数量batch, seq_len, emb_size, 1]
                    self.embedded_chars_expanded,
                    # 卷积核：[卷积核的高度，词向量维度（卷积核的宽度），1（输入通道数），卷积核个数（输出通道数）]
                    filters,
                    # 各维度步长
                    strides=[1, 1, 1, 1],
                    # 'SAME'为等长卷积填充0, 'VALID'为窄卷积不填充
                    padding="VALID",
                    name="conv")
                # relu激活
                hidden = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # 最大池化:https://blog.csdn.net/m0_37586991/article/details/84575325
                pooled = tf.nn.max_pool(
                    # 待池化的四维张量：[batch, height, width, channels]
                    hidden,
                    # 池化窗口大小：长度（大于）等于4的数组，与value的维度对应，一般为[1,height,width,1]，batch和channels上不池化
                    ksize=[1, seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                # 池化后的结果append到pooled_outputs中
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        # 在第3个维度拼接
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, cls_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[cls_num]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            # 线性变换：wx+b
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            # 获取预测结果：最大打分
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            # 求softmax_loss
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

