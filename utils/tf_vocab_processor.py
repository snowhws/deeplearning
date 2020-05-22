#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author wensong

import tensorflow as tf
import numpy as np
import sys
import io
import logging


class TFVocabProcessor(object):
    '''词表处理器
    '''
    def __init__(self,
                 max_document_length,
                 min_frequency=0,
                 tokenizer_fn=None):
        '''初始化

        Args:
            max_document_length: 最长序列长度，不足长度自动padding
            min_frequency: 最小频次，小于等于该值词语抛弃，用UNK替代
            tokenize_fn: 切分函数，输入是字符串，输出是list，yield方式返回
        '''
        self.max_document_length = max_document_length
        self.min_frequency = min_frequency
        # 词表/倒排
        self.vocabulary = {"__PADDING__": 0, "__UNK__": 1}
        self.reverse_vocab = {0: "__PADDING__", 1: "__UNK__"}
        # 词表长度
        self.length = 2
        self.tokenizer_fn = tokenizer_fn
        self.word_freq = {"__PADDING__": -1, "__UNK__": 0}

    def _tokenizer_fn(self, strs):
        '''按空格切分，语料需要事先处理好

        Args:
            strs: 字符串数组
        '''
        for line in strs:
            if line is None:
                yield []
            else:
                yield line.split()

    def feed(self, strs):
        '''词频统计
        '''
        if strs is None:
            return
        # 设置切词函数
        if self.tokenizer_fn is None:
            self.tokenizer_fn = self._tokenizer_fn
        # 统计词频
        for words in self.tokenizer_fn(strs):
            for word in words:
                if word not in self.word_freq:
                    self.word_freq[word] = 1
                else:
                    self.word_freq[word] += 1

    def build(self):
        '''构造词表
        '''
        # 按词频生成id
        for word in self.word_freq:
            if self.word_freq[word] < self.min_frequency:
                continue
            if word not in self.vocabulary:
                self.vocabulary[word] = self.length
                self.reverse_vocab[self.length] = word
                self.length += 1

    def transform(self, strs):
        '''转化成id

        Args:
            strs: 2D array

        Returns:
            np array 2D for word ids
        '''
        ret = []
        for words in self.tokenizer_fn(strs):
            # 先用__PADDING__填充
            ids = [
                self.vocabulary["__PADDING__"]
                for i in range(self.max_document_length)
            ]
            # cut
            for i in range(0, min(len(words), self.max_document_length)):
                if words[i] not in self.vocabulary:
                    ids[i] = self.vocabulary["__UNK__"]
                    self.word_freq["__UNK__"] += 1
                else:
                    ids[i] = self.vocabulary[words[i]]
            ret.append(ids)
        return np.array(ret)

    def save(self, path):
        '''保存词表
        '''
        fw = open(path, "w")
        for wid in sorted(self.reverse_vocab):
            word = self.reverse_vocab[wid]
            freq = self.word_freq[word]
            fw.write(str(wid) + "\t" + word + "\tfreq: " + str(freq) + "\n")
        fw.close()
