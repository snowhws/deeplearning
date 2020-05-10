#!/usr/bin/env python
#-*- coding:utf8 -*-
# @Author: wensong

from ....utils.tf_utils import Utils
import tensorflow as tf


class TFBaseLayer(object):
    '''TF高级层的基类
    '''
    def __init__(self):
        '''基类初始化
        '''
        pass

    def layer(self):
        '''层具体实现交给子类完成
        '''
        raise NotImplementedError
