#!/usr/bin/env python
# coding: utf-8
# @author: wensong

import tensorflow as tf
import numpy as np
from .base_tf_model import BaseTFModel

class TFTextCNN(BaseTFModel):
    '''CNN分类器封装
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    '''
    
    def __init__(self, config, voc_size, pretrain_word_vecs, filter_sizes, num_filters):
        '''TextCNN初始化

        Args:
            config: 词典类型配置
            voc_size: 词表大小
            pretrain_word_vecs: 预训练词向量
            filter_sizes: 所有卷积核的大小，支持多个窗口同时卷积
            num_filters: 卷积核个数
        '''
        # 父类初始化
        BaseTFModel.__init__(self, config, voc_size, pretrain_word_vecs)
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()
        # 参数
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters

    def build_model(self):
        '''build textCNN model
        '''
        # Embedding layer
        with tf.name_scope("word_embedding"):
            if self.pretrain_word_vecs is not None:
                # 设置预训练词向量
                self.embedding = tf.Variable(tf.cast(self.pretrain_word_vecs, 
                                                     dtype=tf.float32, name="pretrain_vec"),
                                             name="embedding")
            else:
                # 使用xavier初始化
                self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.emb_size],
                                              initializer=tf.contrib.layers.xavier_initializer())
            # 查询词嵌入矩阵
            # 将输入词索引转成词向量
            # 输出shape：[batch_size, seq_len, emb_size]
            self.embedded_words = tf.nn.embedding_lookup(embedding, self.input_x)
            # 在-1列扩展一维，tf.nn.conv2d的input参数为四维变量
            # 输出shape: [batch_size, seq_len, emb_size, 1]
            self.embedded_words_expanded = tf.expand_dims(self.embedded_words, -1)
        
        # 所有滤波器/卷积的池化层
        pooled_outputs = [] 
        # 遍历滤波器：可以同时用3、4、5等多个窗口
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # 卷积核shape：[卷积核大小，宽度，输入通数，输出通道数]
                filter_shape = [filter_size, self.emb_size, 1, self.num_filters]
                # 卷积核
                # 随机生成截断正态分布参数，大于两倍标准差stddev即截断
                filters = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filters")
                W = tf.get_variable("W" + str(filter_size), shape=filter_shape,
                                    initializer=tf.truncated_normal_initializer(stddev=0.1))
                b = tf.get_variable("b" + str(filter_size), shape=[self.num_filters],
                                    initializer=tf.zeros_initializer())
                # 卷积
                # 'SAME'为等长卷积填充0, 'VALID'为窄卷积不填充
                conv = tf.nn.conv2d(self.embedded_words_expanded, W,
                                    strides=[1, 1, 1, 1], padding="VALID",
                                    name="conv" + str(filter_size))
                # 非线性变换隐层
                hidden = tf.nn.relu(tf.add(conv, b), name="relu")
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
            
        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        """
        # 获得最后一层LSTM的神经元数量
        hidden_size = self.config["hidden_sizes"][-1]

        # ----------权重：句向量hidden_size维数----------
        # 初始化一个权重向量，是可训练的参数
        W = tf.Variable(tf.random_normal([hidden_size], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        M = tf.tanh(H)

        # ----------self-attention在句向量上来说，Q/K/V都是同一个即H，所以一块求点乘即可：Q*W/K*W/V*W----------
        # 对W和M做矩阵运算，M=[batch_size, time_step, hidden_size]，计算前做维度转换成[batch_size * time_step, hidden_size]
        # newM = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        newM = tf.matmul(tf.reshape(M, [-1, hidden_size]), tf.reshape(W, [-1, 1]))

        # 对newM做维度转换成[batch_size, time_step]
        restoreM = tf.reshape(newM, [-1, self.config["sequence_length"]])

        # ----------softmax归一权重----------

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

