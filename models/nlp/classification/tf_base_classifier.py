#!/usr/bin/env python
#-*- coding:utf8 -*-
# @Author: wensong

import os
import sys
import datetime
import logging
sys.path.append(os.getcwd() + "/../../")
from utils.tf_utils import TFUtils
import tensorflow as tf


class TFBaseClassifier(object):
    '''TF模型基类
    主要实现可继承的公共方法，例如：opt优化器选择、训练、评估等基础函数。
    子类则focus在模型实现上。
    '''
    def __init__(self, flags):
        '''
        Args:
            flags: 全局参数，包含learning_rate、classifier_type等参数
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            pretrain_word_vecs：预训练词向量
        '''
        # 获取全局参数
        self.flags = flags

        # 输入&占位符
        self.input_x = tf.placeholder(tf.int32, [None, None],
                                      name="input_x")  # 输入二维张量
        self.input_y = tf.placeholder(tf.float32, [None, self.flags.cls_num],
                                      name="input_y")  # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")  # 激活概率
        self.pretrain_word_vecs = None  # 预训练语言模型

        # 模型产生的变量
        self.loss = 0.0  # 损失
        self.train_op = None  # 训练器
        self.summary_op = None  # 记录器
        self.summary_writer = None  # 日志写入器
        self.logits = None  # 输出
        self.probability = None  # 预测概率
        self.saver = None  # 保存器: checkpoint模型
        self.predictions = None  # 预测结果
        self.global_step = None  # 全局训练步数

        # 指标metrics
        self.precision = None  # 精确度
        self.recall = None  # 召回率
        self.accuracy = None  # 正确率
        self.f1_score = None  # f1-score

    def build_model(self):
        '''构建模型
        子类实现
        '''
        raise NotImplementedError

    def get_optimizer(self):
        '''获取优化器

        Returns:
            返回优化器
        '''
        optimizer = None

        if self.flags.opt == "adam":
            optimizer = tf.train.AdamOptimizer(self.flags.lr)
        if self.flags.opt == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.flags.lr)
        if self.flags.opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.flags.lr)

        return optimizer

    def add_train_op(self):
        '''训练op
        设置梯度最大截断
        '''
        # 全局训练步数
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # 优化器
        optimizer = self.get_optimizer()
        # 反向求梯度: 等价于compute_gradients
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients,
                                                   self.flags.max_grad_norm)
        # 使用梯度进行优化: minimize等价于先compute_gradients，再apply_gradients
        self.train_op = optimizer.apply_gradients(zip(clip_gradients,
                                                      trainable_params),
                                                  global_step=self.global_step)
        # 记录loss
        tf.summary.scalar("loss", self.loss)
        # 合并scalar的变量
        self.summary_op = tf.summary.merge_all()

    def add_metrics(self):
        '''添加指标统计
        '''
        # 获取结果
        self.get_predictions()
        # 精度
        self.precision, self.precision_update = tf.metrics.precision(
            labels=self.input_y,
            predictions=self.predictions,
            name="precision")
        # 召回
        self.recall, self.recall_update = tf.metrics.recall(
            labels=self.input_y, predictions=self.predictions, name="recall")
        # 正确率
        self.accuracy, self.accuracy_update = tf.metrics.accuracy(
            labels=self.input_y, predictions=self.predictions, name="accuracy")
        # add precision and recall to summary
        tf.summary.scalar('precision', self.precision_update)
        tf.summary.scalar('recall', self.recall_update)
        tf.summary.scalar('accuracy', self.accuracy_update)

    def get_predictions(self):
        '''获取预测结果
        将softmax跑出的概率转为tf.metrics可匹配的预测结果。
        其中，若是multi-label任务，probs中大于0.5的为preds中预测为1的结果。
        若是multi-class任务，则probs中最大值为对应类目的索引，preds中对应值置1。
        '''
        if self.flags.cls_type == "multi-label":
            one = tf.ones_like(self.probability)
            zero = tf.zeros_like(self.probability)
            self.predictions = tf.where(self.probability <= 0.5, x=zero, y=one)
        elif self.flags.cls_type == "multi-class-dense" or \
            self.flags.cls_type == "multi-class-sparse":
            # 取topK
            topk_values, topk_indices = tf.nn.top_k(self.probability,
                                                    k=1,
                                                    sorted=False)
            # 大于等于第k大，True转1.0，False转0.0
            self.predictions = tf.cast(
                tf.greater_equal(self.probability, topk_values), tf.float32)

    def train_onestep(self, sess, x_batch, y_batch):
        '''单步训练

        Args:
            sess: 会话
            x_batch: 输入batch
            y_batch: 标签batch

        Return:
            返回训练记录、损失和预测结果
        '''
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_prob: self.flags.keep_prob
        }
        # 运行会话
        _, summary_op, step, loss, acc = sess.run([
            self.train_op, self.summary_op, self.global_step, self.loss,
            self.accuracy_update
        ],
                                                  feed_dict=feed_dict)
        # 保存log
        self.summary_writer.add_summary(summary_op, step)
        # 打印acc等结果
        time_str = datetime.datetime.now().isoformat()
        logging.info("{}: step {}, loss {:g}, train acc {}".format(
            time_str, step, loss, acc))

    def train(self, sess, graph, vocab_processor, save_path, x_train, y_train,
              x_dev, y_dev):
        '''整体训练入口

        Args:
            sess: 会话
            graph: 整个图
            vocab_processor: 词表处理器
            save_path: 模型保存目录
            x_train: 训练集输入
            y_train: 训练集标签
            x_dev: 评估集输入
            y_dev: 评估集标签
        '''
        # 初始化日志写入器
        timestamp = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.datetime.now())
        log_path = os.path.join(self.flags.log_path, timestamp)
        self.summary_writer = tf.summary.FileWriter(log_path, graph=graph)
        # 模型保存目录
        checkpoint_dir = os.path.abspath(os.path.join(save_path,
                                                      "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 生成batch, zip会将两个数组遍历打包成对元组
        batches = TFUtils.batch_iter(list(zip(x_train,
                                              y_train)), self.flags.batch_size,
                                     self.flags.num_epochs)
        # 自动停止条件
        last_acc = 0.0
        # 按batch训练
        for batch in batches:
            # zip(*)逆向解压
            x_batch, y_batch = zip(*batch)
            # 单步训练
            self.train_onestep(sess, x_batch, y_batch)
            # 获取当前步数
            current_step = tf.train.global_step(sess, self.global_step)
            # 保存模型&词表
            if current_step % self.flags.checkpoint_every == 0:
                path = self.saver.save(sess,
                                       checkpoint_prefix,
                                       global_step=current_step)
                vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))
                logging.info("Saved model&vocab to {}\n".format(path))
            # 评估
            if current_step % self.flags.evaluate_every == 0:
                logging.info("\nEvaluation:")
                curr_acc = self.eval(sess, x_dev, y_dev)
                logging.info("")
                # 不再收敛，则停止
                if abs(curr_acc -
                       last_acc) <= self.flags.acc_convergence_score:
                    logging.info("Accuracy convergenced last_acc: " +
                                 str(last_acc) + " -> curr_acc: " +
                                 str(curr_acc) + ", Stop training!")
                    break
                last_acc = curr_acc

    def eval(self, sess, x_batch, y_batch):
        '''验证模型

        Args:
            sess: 会话
            x_batch: 输入样本
            y_batch: 输入标签
        '''
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.keep_prob: 1.0  # 评估时候通过dropout设置不更新参数
        }

        # 执行会话
        step, loss, accuracy = sess.run(
            [self.global_step, self.loss, self.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        logging.info("{}: step {}, loss {:g}, test acc {:g}".format(
            time_str, step, loss, accuracy))

        return accuracy

    def infer(self, sess, x_batch, y_batch):
        '''验证模型

        Args:
            sess: 会话
            x_batch: 输入样本
            y_batch: 输入标签
        '''
        self.eval(sess, x_batch, y_batch)
