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
        # 参数
        self.filter_sizes = filter_sizes
        self.num_filters = num_filters
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def layer(self):
        '''TextCNN Layer隐层表示

        Returns:
            返回经过TextCNN后的隐层表示
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
                # 最大池化
                pooled = tf.nn.max_pool(hidden,
                    ksize=[1, self.max_seq_len - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool" + str(filter_size))
                # 池化后的结果append起来
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        feature_dim = self.num_filters * len(self.filter_sizes)
        # 在out_channel维度拼接
        h_pool = tf.concat(pooled_outputs, 3)
        h_pool_flat = tf.reshape(h_pool, [-1, feature_dim])
        
        # Add dropout
        with tf.name_scope("dropout_layer"):
            self.h_drop = tf.nn.dropout(h_pool_flat, self.keep_prob)
        
        return self.h_drop

    def build_model(self):
        '''build textCNN model
        '''
        model_emb = self.layer()
        # Final scores and predictions
        with tf.name_scope("fc_output_layer"):
            W = tf.get_variable("W", shape=[num_filters_total, self.cls_num],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b', shape=[self.cls_num],
                                initializer=tf.constant_initializer(0.1))
            self.l2_loss += tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(model_emb, W, b, name="logits")
            self.predictions = tf.nn.softmax(self.logits)
            # self.predictions = self.get_predictions()
        # loss 
        self.loss = self.cal_loss() + self.l2_reg_lambda * self.l2_loss


