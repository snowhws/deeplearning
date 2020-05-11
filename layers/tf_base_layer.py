#!/usr/bin/env python
#-*- coding:utf8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../")
from utils.tf_utils import TFUtils
import tensorflow as tf


class TFBaseLayer(object):
    '''TF高级层的基类
    '''
    def __init__(self):
        '''基类初始化
        '''

    def build(self):
        '''层具体实现交给子类完成
        '''
        raise NotImplementedError
