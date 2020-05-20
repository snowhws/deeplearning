#!/usr/bin/env python
# coding: utf-8
# @author: wensong

import tensorflow as tf
from utils.tf_utils import TFUtils
from tf_base_layer import TFBaseLayer
from tf_ln_layer import TFLNLayer
from tf_soft_att_layer import TFSoftAttLayer


class TFHierarchicalAttLayer(TFBaseLayer):
    '''层次注意力模型
    底层word->sent+attention，上层sent->doc+attention
    '''
    def __init__(self,
                 batched_embedded,
                 max_doc_len,
                 max_seq_len,
                 hidden_size,
                 attention_size,
                 keep_prob,
                 training=True,
                 rnn_type="GRU",
                 scope="hierarchical_attention"):
        '''Hierarchical attention network初始化

        Args:
            batched_embedded: 多篇文档emb表示[Batch, T_sents, T_words, word_emb]
            max_doc_len: doc中最大句子个数
            max_seq_len: 句子里最大词语个数
            hidden_size: BILSTM中每层隐层维数大小
            attention_size: 注意力矩阵宽度
            keep_prob: 多层lstm之间dropout输出时激活概率
            training: 是否训练模式
            rnn_type: 可选择LSTM或GRU
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前layer参数
        self.batched_embedded = batched_embedded
        self.word_emb_size = batched_embedded.get_shape().as_list()[-1]
        self.max_doc_len = max_doc_len
        self.max_seq_len = max_seq_len
        self.hidden_size = hidden_size
        self.att_size = attention_size
        self.keep_prob = keep_prob
        self.training = training
        self.rnn_type = rnn_type
        self.scope = scope

    def build(self):
        '''层次化attention网络

        Returns:
            [Batch,  Hidden_Size]
        '''
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
            # [B, T_s, T_w, emb_size] -> [B * T_s, T_w, emb_size]
            enc = tf.reshape(self.batched_embedded,
                             [-1, self.max_seq_len, self.word_emb_size])

            # sents encoding: [B*T_s, T_w, emb_size] -> [B*T_s, H]
            enc = self._encoding(enc, "sent")

            # split: [B*T_s, H] -> [B, T_s, H]
            enc = tf.reshape(enc, [-1, self.max_doc_len, self.hidden_size])

            # doc encoding: [B, T_s, H] -> [B, H]
            enc = self._encoding(enc, "doc")

            return enc

    def _encoding(self, layer_hidden, scope):
        # lstm encoding
        with tf.variable_scope(scope + "_" + self.rnn_type + "Layer",
                               reuse=tf.AUTO_REUSE):
            # forward LSTM
            fw_lstm_cell = self._rnn_cell(self.hidden_size)
            # backward LSTM
            bw_lstm_cell = self._rnn_cell(self.hidden_size)
            # outputs: (output_fw, output_bw)
            # 其中两个元素的维度都是[B, T, hidden_size],
            outputs, current_state = tf.nn.bidirectional_dynamic_rnn(
                fw_lstm_cell,
                bw_lstm_cell,
                layer_hidden,
                sequence_length=TFUtils.get_sequence_lens(layer_hidden),
                dtype=tf.float32,
                scope=self.rnn_type)

            # 从第三维拼接：[B, T, hidden_size]
            layer_hidden = tf.concat(outputs, 2)

        # 分割成前向和后向的输出
        outputs = tf.split(layer_hidden, num_or_size_splits=2, axis=-1)
        # [B, T, Hidden_Size]
        layer_hidden = outputs[0] + outputs[1]

        # Attention
        layer_hidden = TFSoftAttLayer(layer_hidden, self.att_size,
                                      self.training).build()
        return layer_hidden

    def _rnn_cell(self, hidden_size):
        '''获取RNN Cell
        '''
        rnn_cell = None
        if self.rnn_type == "LSTM":
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        else:
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        # LN归一
        rnn_ln = TFLNLayer(rnn_cell).build()
        # dropout
        rnn_with_dp = tf.nn.rnn_cell.DropoutWrapper(
            rnn_ln, output_keep_prob=self.keep_prob)
        return rnn_with_dp
