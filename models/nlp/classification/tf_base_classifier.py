#!/usr/bin/env python
#-*- coding:utf8 -*-
# @Author: wensong

import os
import sys
import datetime
import logging
import tf_metrics
from tqdm import tqdm
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
        if flags.data_type == "shorttext" or flags.data_type == "longtext_with_title":
            self.input_x = tf.placeholder(
                tf.int32, [None, None],
                name="input_x")  # 输入[Batch, word_id_list]
        if flags.data_type == "longtext_with_title":  # 由句子序列组成的长文本[B, T_s, T_w]
            self.input_c = tf.placeholder(tf.int32, [None, None, None],
                                          name="input_c")
        self.input_y = tf.placeholder(tf.float32, [None, self.flags.cls_num],
                                      name="input_y")  # 标签[Batch, class_num]
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

    def add_metrics(self):
        '''添加指标统计
        '''
        # 全局训练步数
        self.global_step = tf.Variable(0, name="global_step", trainable=False)
        # 获取结果
        self.get_predictions()
        # 精度
        self.precision, self.precision_update = tf_metrics.precision(
            self.input_y,
            self.predictions,
            self.flags.cls_num, [i for i in range(self.flags.cls_num)],
            average="micro")
        # 召回
        self.recall, self.recall_update = tf_metrics.recall(
            self.input_y,
            self.predictions,
            self.flags.cls_num, [i for i in range(self.flags.cls_num)],
            average="micro")
        # 准确率
        self.accuracy, self.accuracy_update = tf.metrics.accuracy(
            labels=self.input_y, predictions=self.predictions, name="accuracy")
        # add precision and recall to summary
        tf.summary.scalar('precision', self.precision_update)
        tf.summary.scalar('recall', self.recall_update)
        tf.summary.scalar('accuracy', self.accuracy_update)
        # 记录loss
        tf.summary.scalar("loss", self.loss)
        # 合并scalar的变量
        self.summary_op = tf.summary.merge_all()

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

    def train_onestep(self, sess, train_tuple, pbar):
        '''单步训练

        Args:
            sess: 会话
            train_tuple: 训练样本(x, ..., y)
            pbar: tqdm进度条

        Return:
            返回训练记录、损失和预测结果
        '''
        feed_dict = {}
        if self.flags.data_type == "shorttext":
            feed_dict = {
                self.input_x: train_tuple[0],
                self.input_y: train_tuple[1],
                self.keep_prob: self.flags.keep_prob
            }
        elif self.flags.data_type == "longtext_with_title":
            feed_dict = {
                self.input_x: train_tuple[0],
                self.input_c: train_tuple[1],
                self.input_y: train_tuple[2],
                self.keep_prob: self.flags.keep_prob
            }
        # 运行会话
        _, step, loss, acc = sess.run(
            [self.train_op, self.global_step, self.loss, self.accuracy_update],
            feed_dict=feed_dict)
        # 进度条右侧打印loss和acc信息
        pbar.set_postfix(loss=loss, acc=acc)

        return step

    def train(self, sess, graph, vocab_processor, save_path, train_tuple,
              test_tuple):
        '''整体训练入口

        Args:
            sess: 会话
            graph: 整个图
            vocab_processor: 词表处理器
            save_path: 模型保存目录
            train_tuple: 训练集
            test_tuple: 评估集
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
        batches = TFUtils.batch_iter(train_tuple, self.flags.batch_size,
                                     self.flags.num_epochs)
        n_batches = int((len(train_tuple[0]) - 1) / self.flags.batch_size) + 1
        # Epoch训练进度条
        e_pbar = tqdm(total=self.flags.num_epochs)
        e_pbar.set_description("Progress of Epoches")
        # Batch训练进度条
        b_pbar = tqdm(total=n_batches)
        b_pbar.set_description("Progress of Batches")
        # 自动停止条件
        last_acc = 0.0
        # 按batch训练
        idx = 0
        for batch in batches:
            # zip(*)逆向解压
            batch_tuple = zip(*batch)
            # 单步训练
            current_step = self.train_onestep(sess, batch_tuple, b_pbar)
            # 进度条update
            if idx % n_batches == 0:
                b_pbar = tqdm(total=n_batches)
                b_pbar.set_description("Progress of Batches")
                e_pbar.update(1)
            b_pbar.update(1)
            # 保存模型&词表
            if current_step % self.flags.checkpoint_every == 0:
                path = self.saver.save(sess,
                                       checkpoint_prefix,
                                       global_step=current_step)
                TFUtils.save_vocab(vocab_processor,
                                   os.path.join(checkpoint_dir, "vocab"))
                logging.info("Saved model&vocab to {}\n".format(path))
            # 评估
            if current_step % self.flags.evaluate_every == 0:
                curr_acc = self.eval(sess, test_tuple)
                # 不再收敛，则停止
                if abs(curr_acc -
                       last_acc) <= self.flags.acc_convergence_score:
                    logging.info("Accuracy convergenced last_acc: " +
                                 str(last_acc) + " -> curr_acc: " +
                                 str(curr_acc) + ", Stop training!")
                    break
                last_acc = curr_acc
            idx += 1
        # 进度条关闭
        b_pbar.close()
        e_pbar.close()

    def eval(self, sess, test_tuple):
        '''验证模型

        Args:
            sess: 会话
            test_tuple: 测试样例(x, ..., y)
        '''
        feed_dict = {}
        if self.flags.data_type == "shorttext":
            feed_dict = {
                self.input_x: test_tuple[0],
                self.input_y: test_tuple[1],
                self.keep_prob: 1.0  # 评估时dropout关闭
            }
        elif self.flags.data_type == "longtext_with_title":
            feed_dict = {
                self.input_x: test_tuple[0],
                self.input_c: test_tuple[1],
                self.input_y: test_tuple[2],
                self.keep_prob: 1.0  # 评估时dropout关闭
            }

        # 执行会话
        logging.info("\nEvaluation:")
        summary_op, step, loss, accuracy = sess.run([
            self.summary_op, self.global_step, self.loss, self.accuracy_update
        ], feed_dict)

        # 保存log
        self.summary_writer.add_summary(summary_op, step)

        logging.info("loss {:g}, test acc {:g}\n".format(loss, accuracy))

        return accuracy

    def infer(self, sess, test_tuple):
        '''验证模型

        Args:
            sess: 会话
        '''
        self.eval(sess, test_tuple)
