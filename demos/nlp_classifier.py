#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../")
from utils.tf_utils import TFUtils
import tensorflow as tf
import numpy as np
import time
import datetime
import logging
from executes.executor import Executor
from executes.init_processor import InitProcessor
from executes.pre_processor import PreProcessor
from executes.graph_processor import GraphProcessor
from executes.session_processor import SessionProcessor

# 设定日志级别和格式
logging.basicConfig(
    level=logging.DEBUG,
    format=
    '%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

FLAGS = None


def get_flags():
    '''获取脚本输入参数
    '''
    # 当前路径
    current_path = os.getcwd()
    # 任务名: TextCNN/BILSTMAtt
    tf.flags.DEFINE_string(
        "task_name", "TextCNN",
        "task name, canbe BILSTMAtt/TextCNN(default: TextCNN)")
    tf.flags.DEFINE_boolean("training", True, "True or False(default: True)")
    # 样本集合相关参数
    tf.flags.DEFINE_float(
        "dev_sample_percentage", .1,
        "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string(
        "data_file",
        os.path.join(current_path, "../corpus/nlp/english/rt-polarity.all"),
        "Data source for training and testing.(default: ../corpus/nlp/english/rt-polarity.all)"
    )
    tf.flags.DEFINE_string(
        "data_type", "shorttext",
        "Data Type, shorttext/longtext_with_title(default: shorttext)")

    # 分类器公共参数
    tf.flags.DEFINE_integer("cls_num", 2, "size of classes(default: 2)")
    tf.flags.DEFINE_string(
        "cls_type", "multi-class-dense",
        "Classifier Type, multi-label/multi-class-sparse/multi-class-dense(default: multi-class-dense)"
    )
    tf.flags.DEFINE_integer(
        "vocab_size", 100000,
        "Size of vocabulary dict, will auto-update after PreProcessor(default: 100000)"
    )
    tf.flags.DEFINE_integer("emb_size", 128,
                            "Dimensionality of word embedding (default: 128)")
    tf.flags.DEFINE_integer("max_seq_len", 128,
                            "max len of input seq(default: 128)")
    tf.flags.DEFINE_integer("max_doc_len", 50,
                            "max sents num of a document(default: 50)")
    tf.flags.DEFINE_boolean(
        "word_emb_trainable", True,
        "pretrain word embedding trainable(default: True)")

    # TextCNN相关参数
    tf.flags.DEFINE_string("filter_sizes", "2,3,4,5",
                           "TextCNN filter sizes (default: '2,3,4,5')")
    tf.flags.DEFINE_integer(
        "num_filters", 128, "Number of filters per filter size (default: 128)")

    # BILSTMATT相关参数
    tf.flags.DEFINE_string(
        "hidden_sizes", "128",
        "BILSTM hidden sizes, including MultiLayers, can be 128,256,... (default: '128')"
    )
    tf.flags.DEFINE_integer("attention_size", 128,
                            "BILSTM-Attention size(default: 128)")

    # Transformer等模型相关参数
    tf.flags.DEFINE_integer("hidden_size", 1024,
                            "hidden size for Transformer FFN (default: 1024)")
    tf.flags.DEFINE_integer(
        "num_blocks", 1, "num of blocks of Transformer Encoder(default: 1)")

    # 训练相关参数
    tf.flags.DEFINE_float("lr", 5e-3, "learning rate(default: 5e-3)")
    tf.flags.DEFINE_integer("decay_steps", 100,
                            "decay steps for learning_rate(default: 100)")
    tf.flags.DEFINE_float("decay_rate", 0.9,
                          "decay_rate for learning_rate(default: 0.9)")
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 100,
                            "Number of training epochs (default: 100)")
    tf.flags.DEFINE_float("keep_prob", 0.1,
                          "Dropout keep probability (default: 0.1)")
    tf.flags.DEFINE_float("l2_reg_lambda", 3e-3,
                          "L2 regularization lambda (default: 3e-3)")
    tf.flags.DEFINE_integer(
        "evaluate_every", 100,
        "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100,
                            "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5,
                            "Number of checkpoints to store (default: 5)")
    tf.flags.DEFINE_string("opt", "adam", "Optimizer name(default: adam)")
    tf.flags.DEFINE_float("max_grad_norm", 5.0,
                          "Max Gradient Norm(default: 5.0)")
    tf.flags.DEFINE_float(
        "acc_convergence_score", 1e-4,
        "Convergence condition by abs(current_acc - last_acc) < acc_convergence_score(default: 1e-4)"
    )

    #  设备及日志相关
    tf.flags.DEFINE_boolean(
        "allow_soft_placement", True,
        "Allow device soft device placement")  # 如果指定设备不存在，tf自动分配设备
    tf.flags.DEFINE_boolean("log_device_placement", False,
                            "Log placement of ops on devices")  # 是否打印备份日志
    tf.flags.DEFINE_string("save_path", "save_models",
                           "save_path (default: 'save_models')")
    tf.flags.DEFINE_string("log_path", os.path.join(current_path, "../log/"),
                           "summary path for tensorboard(default: '../log')")

    return tf.flags.FLAGS


def main(argv=None):
    # 定义执行器
    exe = Executor(FLAGS)
    # 添加processor
    exe.add_processor(InitProcessor())  # 参数初始化
    exe.add_processor(PreProcessor())  # 样本预处理
    exe.add_processor(GraphProcessor())  # 构建模型
    exe.add_processor(SessionProcessor())  # 创建会话
    # 执行
    exe.run()


if __name__ == '__main__':
    '''运行
    '''
    # 获取命令行参数
    FLAGS = get_flags()
    tf.app.run()
