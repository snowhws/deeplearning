#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
from utils.tf_utils import TFUtils
import tensorflow as tf
from tf_base_layer import TFBaseLayer


class TFClassifierLayer(TFBaseLayer):
    '''分类层
    1、分训练(train)模式与预测(infer)模式，训练模式会计算与返回loss
    2、支持Multi-Class-Dense&Multi-Class-Sparse&Multi-label
    3、支持加dropout层，大部分模型最后会带一层dropout layer
    '''
    def __init__(self,
                 training,
                 in_hidden,
                 cls_num,
                 cls_type,
                 input_y,
                 keep_prob,
                 l2_reg_lambda,
                 scope="classifier"):
        '''初始化

        Args:
            training: 是否在训练，训练模式会返回loss
            in_hidden: 输入层
            cls_num: 类目数
            cls_type: multi-class-dense/multi-class-sparse/multi-label
            input_y: 标签labels, shape [batch, dense/sparse/one-hot labels]
            l2_reg_lambda: l2 reg lambda prob
        '''
        TFBaseLayer.__init__(self)
        self.training = training
        self.in_hidden = in_hidden
        self.hidden_size = in_hidden.get_shape()[-1]
        self.cls_num = cls_num
        self.cls_type = cls_type
        self.input_y = input_y
        self.keep_prob = keep_prob
        self.l2_reg_lambda = l2_reg_lambda
        self.scope = scope

    def build(self):
        '''分类层，分训练模式与推理模式

        Returns:
            probability: 预测概率
            logits: output层
            loss: 损失
        '''
        # 定义l2损失
        l2_loss = tf.constant(0.0)
        # fc layer
        probability = None
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # add dropout before classify layer
            dropout_layer = tf.layers.dropout(self.in_hidden,
                                              1.0 - self.keep_prob,
                                              training=self.training)
            W = tf.get_variable(
                "W",
                shape=[self.hidden_size, self.cls_num],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('b',
                                shape=[self.cls_num],
                                initializer=tf.constant_initializer(0.1))
            # logits: [B, cls_num]
            self.logits = tf.nn.xw_plus_b(dropout_layer, W, b, name="logits")
            # softmax or sigmoid
            if self.cls_type == "multi-label":
                probability = tf.nn.sigmoid(self.logits)
            else:
                probability = tf.nn.softmax(self.logits)
            # 训练模式
            if self.training:
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                self.loss = self._cal_loss() + self.l2_reg_lambda * l2_loss

        return probability, self.logits, self.loss

    def _scale_l2(x, norm_length):
        '''按l2缩放x
        '''
        # shape(x) = (batch, num_timesteps, d)
        # Divide x by max(abs(x)) for a numerically stable L2 norm.
        # 2norm(x) = a * 2norm(x/a)
        # Scale over the full sequence, dims (1, 2)
        alpha = tf.reduce_max(tf.abs(x), (1, 2), keepdims=True) + 1e-12
        l2_norm = alpha * tf.sqrt(
            tf.reduce_sum(tf.pow(x / alpha, 2), (1, 2), keepdims=True) + 1e-6)
        x_unit = x / l2_norm
        return norm_length * x_unit

    def _add_perturbation(self, embedded, loss, epsilon=5):
        """在emedding上添加扰动，增强鲁棒性.
        参考Paper：《ADVERSARIAL TRAINING METHODS FOR SEMI-SUPERVISED TEXT CLASSIFICATION》
        Adds gradient to embedding and recomputes classification loss.
        """
        grad, = tf.gradients(
            loss,
            embedded,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N)
        grad = tf.stop_gradient(grad)
        perturb = self._scale_l2(grad, epsilon)
        return embedded + perturb

    def _cal_loss(self):
        '''计算损失
        支持Mult-Label和Multi-Class

        Returns:
            返回loss均值
        '''
        # 训练时平滑标签
        labels = self._label_smoothing(self.input_y)
        with tf.name_scope("loss"):
            losses = 0.0
            if self.cls_type == "multi-label":
                # 多个二分类sigmoid实现multi-label
                # input_y: [batch, dense labels]
                losses = tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.logits, labels=labels)
            elif self.cls_type == "multi-class-dense":
                # label为稠密概率分布
                # input_y: [batch, dense labels]
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=labels)
            elif self.cls_type == "multi-class-sparse":
                # one-hot类型稀疏表示标签
                labels = tf.argmax(self.input_y, axis=1)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.logits, labels=labels)
            # 一个batch内loss取均值
            loss = tf.reduce_mean(losses)

            return loss

    def _label_smoothing(self, inputs, epsilon=0.1):
        '''标签平滑技术，提升泛化能力
        '''
        V = inputs.get_shape().as_list()[-1]  # number of channels/classes
        return ((1 - epsilon) * inputs) + (epsilon / V)
