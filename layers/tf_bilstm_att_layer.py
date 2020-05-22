#!/usr/bin/env python
# coding: utf-8
# @author: wensong

import tensorflow as tf
from utils.tf_utils import TFUtils
from tf_base_layer import TFBaseLayer
from tf_ln_layer import TFLNLayer
from tf_soft_att_layer import TFSoftAttLayer


class TFBILSTMAttLayer(TFBaseLayer):
    '''多层bi-lstm加attention层封装
    底层可以多个双向lstm，顶层是SoftAttention加权隐层表示。
    '''
    def __init__(self,
                 in_hidden,
                 hidden_sizes,
                 attention_size,
                 keep_prob,
                 training=True,
                 rnn_type="GRU",
                 scope="bilstm_attention"):
        '''Bi-LSTM-ATTENTION初始化

        Args:
            in_hidden: 输入层
            hidden_sizes: 多层BILSTM中每层隐层维数大小
            attention_size: 注意力矩阵宽度
            keep_prob: 多层lstm之间dropout输出时激活概率
            training: 是否训练模式
            rnn_type: 可选择LSTM或GRU
        '''
        # 父类初始化
        TFBaseLayer.__init__(self)
        # 当前layer参数
        self.in_hidden = in_hidden
        self.hidden_sizes = hidden_sizes
        self.att_size = attention_size
        self.keep_prob = keep_prob
        self.training = training
        self.rnn_type = rnn_type
        self.scope = scope

    def build(self):
        '''多层bilstm-attention Layer隐层表示

        Returns:
            返回经过BILSTM-ATTENTION后的隐层表示，shape为[Batch, last_Hidden_Size]
        '''
        layer_hidden = self.in_hidden
        # n个双层lstm
        for idx, hidden_size in enumerate(self.hidden_sizes):
            with tf.variable_scope(self.scope + "_" + self.rnn_type +
                                   "Layer_" + str(idx),
                                   reuse=tf.AUTO_REUSE):
                # forward LSTM
                fw_lstm_cell = self._rnn_cell(hidden_size)
                # backward LSTM
                bw_lstm_cell = self._rnn_cell(hidden_size)
                # dynamic lens
                seq_lens = TFUtils.get_sequence_lens(layer_hidden)
                tf.summary.histogram(self.scope + "_" + str(idx) + "_seqlens",
                                     seq_lens)
                # LN归一
                ln = TFLNLayer(layer_hidden).build()
                # outputs: (output_fw, output_bw)
                # 其中两个元素的维度都是[batch_size, max_time, hidden_size],
                outputs, current_state = tf.nn.bidirectional_dynamic_rnn(
                    fw_lstm_cell,
                    bw_lstm_cell,
                    ln,
                    sequence_length=seq_lens,
                    dtype=tf.float32,
                    scope=self.rnn_type + str(idx))

                # 从第三维拼接：[batch_size, time_step, hidden_size]
                layer_hidden = tf.concat(outputs, 2)

        # 分割成前向和后向的输出
        outputs = tf.split(layer_hidden, num_or_size_splits=2, axis=-1)
        # [Batch, TimeStep, last_Hidden_Size]
        bilstm_layer = outputs[0] + outputs[1]

        # Attention
        output = TFSoftAttLayer(bilstm_layer, self.att_size,
                                self.training).build()

        # [Batch, last_Hidden_Size]
        return output

    def _rnn_cell(self, hidden_size):
        '''获取RNN Cell
        '''
        rnn_cell = None
        if self.rnn_type == "LSTM":
            rnn_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size)
        else:
            rnn_cell = tf.nn.rnn_cell.GRUCell(num_units=hidden_size)
        # dropout
        rnn_with_dp = tf.nn.rnn_cell.DropoutWrapper(
            rnn_cell, output_keep_prob=self.keep_prob)
        return rnn_with_dp
