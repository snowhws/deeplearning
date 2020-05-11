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


def main(argv=None):
    # 定义执行器
    exe = Executor()
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
    tf.flags.DEFINE_integer("max_seq_len", 1024,
                            "max len of input seq(default: 1024)")
    tf.app.run()
    # tf.compat.v1.app.run()
