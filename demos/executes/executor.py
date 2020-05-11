#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
import logging


class Executor(object):
    '''执行器：执行器用于串联一系列processor，从而运行核心程序
    '''
    def __init__(self):
        '''初始化：执行哪个任务由task_name指定
        '''
        # processor 返回值
        self.returns = {}
        # processor list
        self.pro_list = []

    def add_processor(self, processor):
        '''添加一个processor
        '''
        self.pro_list.append(processor)

    def run(self):
        '''按顺序执行所有processor
        '''
        for processor in self.pro_list:
            logging.info(processor.name + " processor started.")
            # 执行并存储返回值，返回值同时也是下一个processor的输入参数
            self.returns[processor.name] = processor.execute(self.returns)
            logging.info(processor.name + " processor finished.")
