#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import logging
import tensorflow as tf
import numpy as np


class InitProcessor(object):
    '''初始化处理器
    '''
    def __init__(self):
        self.name = "INIT"

    def execute(self, params):
        '''初始化参数，使用tf的flags定义
        '''
        # 获取参数
        flags = params["FLAGS"]
        # AdamW: 不需要L2正则，优化器直接会做Weight Decay
        if flags.opt == "adamw":
            flags.l2_reg_lambda = 0.0
            logging.info(
                "Adam/AdamW optimizer, then l2_reg_lambda reset to 0.0")

        return flags
