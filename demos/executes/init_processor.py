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
        flags = params["FLAGS"]
        # TODO: do something to flags

        return flags
