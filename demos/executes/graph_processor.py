#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../../")
import tensorflow as tf
import numpy as np
from models.nlp.classification.tf_bilstmatt_classifier import TFBILSTMATTClassifier
from models.nlp.classification.tf_textcnn_classifier import TFTextCNNClassifier
from models.nlp.classification.tf_transformer_classifier import TFTransformerClassifier


class GraphProcessor(object):
    '''构图处理器：构造Graph
    '''
    def __init__(self):
        self.name = "GRAPH"

    def execute(self, params):
        '''构造Graph
        '''
        # 获取前序processor结果
        flags = params["INIT"]
        x_train, y_train, vocab_processor, x_dev, y_dev = params["PRE"]
        # 创建图
        graph = tf.Graph()
        # 构图
        with graph.as_default():
            # 创建模型
            model = None
            if flags.task_name == "TextCNN":
                model = TFTextCNNClassifier(flags).build_model()
            elif flags.task_name == "BILSTMAtt":
                model = TFBILSTMATTClassifier(flags).build_model()
            elif flags.task_name == "Transformer":
                model = TFTransformerClassifier(flags).build_model()
            # 添加统计指标
            model.add_metrics()
            # 添加训练优化器等
            if flags.mode == "train":
                model.add_train_op()

        return model, graph
