#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author wensong

import tensorflow as tf
import numpy as np

class TFUtils(object):
    '''工具类
    '''
    
    @staticmethod
    def default_dict(dct, key, dft_value):
        '''根据key返回dict内的value，否则返回默认value
        '''        
        if key not in dct:
            return dft_value

        return dct[key]

    @staticmethod
    def nonzero_indices(inputs):
        '''获取张量非零索引

        Returns:
            返回索引张量
        '''        
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(inputs, zero)
        indices = tf.where(where)
        
        return indices



