#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import tensorflow as tf
from tf_base_layer import TFBaseLayer


class TFEmbeddingLayer(TFBaseLayer):
    '''word->embedding层的封装，支持传入预训练word_emb
    '''
    def __init__(self,
                 input_x,
                 vocab_size,
                 emb_size,
                 pretrain_word_vecs=None,
                 word_emb_trainable=True):
        '''初始化

        Args:
            input_x: 词序列的one-hot表示, shape [batch, wordid_list]
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            emb_size: 词向量维数
            pretrain_word_vecs: 预训练词向量
            word_emb_trainable: 预训练词向量是否可update
        '''
        TFBaseLayer.__init__(self)
        self.input_x = input_x
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.pretrain_word_vecs = pretrain_word_vecs
        self.word_emb_trainable = word_emb_trainable

    def build(self):
        '''embedding layer
        '''
        # 词嵌入层
        with tf.name_scope("word_embedding"):
            embedding = None
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.pretrain_word_vecs is not None:
                embedding = tf.get_variable(
                    initializer=self.pretrain_word_vecs,
                    trainable=self.word_emb_trainable,
                    name="embedding")
            else:
                embedding = tf.get_variable(
                    "embedding",
                    shape=[self.vocab_size, self.emb_size],
                    initializer=tf.contrib.layers.xavier_initializer())
            # 查询词嵌入矩阵
            # 将输入词索引转成词向量
            # 输出shape：[batch_size, seq_len, emb_size]
            self.output = tf.nn.embedding_lookup(embedding, self.input_x)

            return self.output
