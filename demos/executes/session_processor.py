#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import datetime
import sys
sys.path.append(os.getcwd() + "/../../")
import tensorflow as tf
import numpy as np
from models.nlp.classification.tf_bilstmatt_classifier import TFBILSTMATTClassifier
from models.nlp.classification.tf_textcnn_classifier import TFTextCNNClassifier


class SessionProcessor(object):
    '''会话处理器
    '''
    def __init__(self):
        self.name = "SESSION"
        self.save_path = ""

    def execute(self, params):
        '''启动模型
        '''
        # 获取前序processor结果
        flags = params["INIT"]
        model, graph = params["GRAPH"]
        train_tuple, vocab_processor, test_tuple = params["PRE"]

        # 保存目录
        timestamp = "{0:%Y-%m-%d_%H-%M-%S/}".format(datetime.datetime.now())
        self.save_path = os.path.abspath(
            os.path.join(os.path.curdir, "../" + flags.save_path, timestamp))
        # session配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=flags.allow_soft_placement,
            log_device_placement=flags.log_device_placement)
        # 创建session
        with tf.Session(config=session_conf, graph=graph) as sess:
            # 保存器
            model.saver = tf.train.Saver(tf.global_variables(),
                                         max_to_keep=flags.num_checkpoints)
            # 变量initial
            sess.run(tf.global_variables_initializer())
            # 重置accuracy两个局部变量accuracy/count
            # 会使得后续train中accuracy的统计是累加的
            # 如果在train中按batch reset，则acc统计的是每个batch的acc
            sess.run(tf.local_variables_initializer())
            # 训练or预测
            if flags.training:
                model.train(sess, graph, vocab_processor, self.save_path,
                            train_tuple, test_tuple)
            else:
                model.infer()
            # 关闭session节省资源
            sess.close()

        return None
