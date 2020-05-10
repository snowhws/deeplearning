#!/usr/bin/env python
#-*- coding:utf8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../../")
from utils.tf_utils import TFUtils
import tensorflow as tf


class TFBaseClassifier(object):
    '''TF模型基类
    主要实现可继承的公共方法，例如：opt优化器选择、训练、评估等基础函数。
    子类则focus在模型实现上。
    '''
    def __init__(self, config, vocab_size=None, pretrain_word_vecs=None):
        '''
        Args:
            config: 模型配置参数（词典类型），包含learning_rate、classifier_type等参数
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            pretrain_word_vecs：预训练词向量
            pretrain_word_vecs 和 vocab_size必须有一个不为None
        '''
        # config params
        self.config = config  # 词典结构的配置
        self.mode = TFUtils.default_dict(self.config, "mode",
                                         "train")  # 模式：infer or train
        self.lr = TFUtils.default_dict(self.config, "learning_rate",
                                       1e-4)  # 学习率
        self.emb_size = TFUtils.default_dict(self.config, "emb_size",
                                             128)  # embedding维数
        self.word_emb_trainable = TFUtils.default_dict(self.config,
                                                       "word_emb_trainable",
                                                       True)  # 词向量是否可训练
        self.cls_num = TFUtils.default_dict(self.config, "cls_num", 2)  # 类目个数
        self.max_seq_len = TFUtils.default_dict(self.config, "max_seq_len",
                                                1024)  # 句子最大长度
        self.cls_type = TFUtils.default_dict(self.config, "classifier_type",
                                             "multi-class-dense")  # 分类器类型
        self.opt = TFUtils.default_dict(self.config, "optimization",
                                        "adam")  # 优化器
        self.max_grad_norm = TFUtils.default_dict(self.config, "max_grad_norm",
                                                  5.0)  # 梯度截取率
        self.l2_reg_lambda = TFUtils.default_dict(self.config, "l2_reg_lambda",
                                                  0.0)  # l2正则比例

        # model params
        self.vocab_size = vocab_size  # 词表大小
        self.pretrain_word_vecs = pretrain_word_vecs  # 支持预训练词向量
        self.input_x = tf.placeholder(tf.int32, [None, None],
                                      name="input_x")  # 输入二维张量
        self.input_y = tf.placeholder(tf.float32, [None, self.cls_num],
                                      name="input_y")  # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # 激活概率

        # training params
        self.l2_loss = tf.constant(0.0)  # 定义l2损失
        self.loss = 0.0  # 损失
        self.train_op = None  # 训练器
        self.summary_op = None  # 记录
        self.logits = None  # 输出
        self.results = None  # 预测结果
        self.saver = None  # 保存器: checkpoint模型

    def build_model(self):
        '''构建模型
        子类实现
        '''
        raise NotImplementedError

    def get_optimizer(self, lr=1e-5):
        '''获取优化器

        Returns:
            返回优化器
        '''
        optimizer = None

        if self.opt == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        if self.opt == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        if self.opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        return optimizer

    def get_train_op(self):
        '''训练op
        设置梯度最大截断

        Returns:
            返回train_op以及summary_op
        '''
        # 优化器
        optimizer = self.get_optimizer()

        # 反向求梯度
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   self.max_grad_norm)
        # 使用梯度进行优化
        train_op = optimizer.apply_gradients(
            zip(clip_gradients, trainable_params))
        # 记录loss
        tf.summary.scalar("loss", self.loss)
        # 自动管理记录
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_results(self, predictions):
        '''获取预测结果

        Return:
            返回预测结果，结果是列表形态。

            若是multi-label任务，则结果是包含每个类目概率的数组。
            若是multi-class任务，则结果是第几个类的索引，放在数组中。
        '''
        results = None

        if self.cls_type == "multi-label":
            results = tf.cast(predictions, tf.float32)
        elif self.cls_type == "multi-class-dense" or \
            self.cls_type == "multi-class-sparse":
            results = tf.argmax(predictions, axis=1)
        return results

    def init_saver(self):
        '''初始化saver对象
        '''
        if self.mode == "infer":
            return

        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, keep_prob):
        '''训练

        Args:
            sess: 会话
            batch: batch数据
            keep_prob: 激活概率

        Return:
            返回训练记录、损失和预测结果
        '''
        feed_dict = {
            self.input_x: batch["x"],
            self.input_y: batch["y"],
            self.keep_prob: keep_prob
        }

        # 运行会话
        _, summary, loss, predictions = sess.run(
            [self.train_op, self.summary_op, self.loss], feed_dict=feed_dict)

        return summary, loss

    def accuracy(self):
        '''
        '''
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions,
                                           tf.argmax(self.input_y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"),
                                      name="accuracy")

            return accuracy

    def eval(self, sess, batch):
        '''验证
        Args:
            sess: 会话
            batch: batch数据

        Return:
            记录、损失和预测结果
        '''
        feed_dict = {
            self.input_x: batch["x"],
            self.input_y: batch["y"],
            self.keep_prob: 1.0
        }  # 激活概率为1.0，使得dropout层失效

        summary, loss, predictions = sess.run(
            [self.summary_op, self.loss, self.predictions],
            feed_dict=feed_dict)
        return summary, loss, predictions

    def infer(self, sess, inputs):
        '''预测

        Args:
            sess: 会话
            inputs: batch数据

        Return:
            预测结果
        '''
        feed_dict = {self.input_x: inputs, self.keep_prob: 1.0}

        predicts = sess.run(self.predictions, feed_dict=feed_dict)

        return predicts
