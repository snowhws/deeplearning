#!/usr/bin/env python
#-*- coding:utf8 -*-
from ....utils.tf_utils import Utils
import tensorflow as tf

class BaseTFModel(object):
    '''TF模型基类
    主要实现可继承的公共方法，例如：loss计算、opt优化器选择、训练、评估等基础函数。
    子类则focus在模型实现上。
    '''

    def __init__(self, config, vocab_size=None, word_vectors=None):
        """
        Args:
            config: 模型配置参数（词典类型），包含learning_rate、classifier_type等参数
            vocab_size: 词向量为空时，使用vocab_size来初始化词向量
            word_vectors：预训练好的词向量
            word_vectors 和 vocab_size必须有一个不为None
        """
        self.config = config                                                            # 词典结构的配置
        self.vocab_size = vocab_size                                                    # 词表大小
        self.word_vectors = word_vectors                                                # 支持预训练词向量                    
        self.input_x = tf.placeholder(tf.int32, [None, None], name="input_x")           # 输入二维张量
        self.input_y = tf.placeholder(tf.float32,                                       
                                      [None, self.config["num_classes"]], 
                                      name="input_y")                                   # 标签
        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")                   # 激活概率

        self.l2_loss = tf.constant(0.0)                                                 # 定义l2损失
        self.loss = 0.0                                                                 # 损失
        self.train_op = None                                                            # 训练器
        self.summary_op = None                                                          # 记录
        self.logits = None                                                              # 输出
        self.predictions = None                                                         # 预测结果
        self.saver = None                                                               # 保存器: checkpoint模型
        
        # configs
        self.lr = Utils.default_dict(self.config, "learning_rate", 1e-4)                # 学习率
        self.cls_type = Utils.default_dict(self.config, "classifier_type",
                                           "multi-class-dense")                         # 分类器类型
        self.opt = Utils.default_dict(self.config, "optimization", "adam")              # 优化器 
        self.max_grad_norm = Utils.default_dict(self.config, 
                                                "max_grad_norm", 5.0)                   # 梯度截取率 

    def cal_loss(self):
        """计算损失
        支持Mult-Label和Multi-Class
        
        Returns:
            返回loss均值
        """
        with tf.name_scope("loss"):
            losses = 0.0
            if self.cls_type == "multi-label":
                # 多个二分类sigmoid实现multi-label
                losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits,
                                                                labels=self.input_y)
            elif self.cls_type == "multi-class-dense":
                # label为稠密概率分布
                losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, 
                                                                 labels=self.input_y)
            elif self.cls_type == "multi-class-sparse":
                # label为稀疏标签
                labels = Utils.nonzero_indices(tf.input_y)
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits,
                                                                        labels=labels[0])
            # 均值计算
            loss = tf.reduce_mean(losses)                                                               

            return loss

    def get_optimizer(self, lr=1e-5):
        """获取优化器
        
        Returns:
            返回优化器
        """ 
        optimizer = None

        if self.opt == "adam":
            optimizer = tf.train.AdamOptimizer(self.lr)
        if self.opt == "rmsprop":
            optimizer = tf.train.RMSPropOptimizer(self.lr)
        if self.opt == "sgd":
            optimizer = tf.train.GradientDescentOptimizer(self.lr)

        return optimizer

    def get_train_op(self):
        """训练op
        设置梯度最大截断

        Returns:
            返回train_op以及summary_op
        """
        # 优化器
        optimizer = self.get_optimizer()

        # 反向求梯度
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        # 梯度截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_grad_norm)
        # 使用梯度进行优化
        train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
        # 记录loss
        tf.summary.scalar("loss", self.loss)
        # 自动管理记录
        summary_op = tf.summary.merge_all()

        return train_op, summary_op

    def get_predictions(self):
        """获取预测结果
        
        Return:
            返回预测结果，结果是列表形态。

            若是multi-label任务，则结果是包含每个类目概率的数组。
            若是multi-class任务，则结果是第几个类的索引，放在数组中。
        """
        predictions = None

        if self.cls_type == "multi-label":
            predictions = tf.cast(self.logits, tf.float32, name="predictions")
        elif self.cls_type == "multi-class-dense" or \
            self.cls_type == "multi-class-sparse":
            predictions = tf.argmax(self.logits, axis=0, name="predictions")
        return predictions

    def build_model(self):
        """创建模型: 子类实现
        """
        raise NotImplementedError

    def init_saver(self):
        """初始化saver对象
        """
        self.saver = tf.train.Saver(tf.global_variables())

    def train(self, sess, batch, keep_prob):
        """训练

        Args:
            sess: 会话
            batch: batch数据
            keep_prob: 激活概率
        
        Return: 
            返回训练记录、损失和预测结果
        """
        feed_dict = {self.input_x: batch["x"],
                     self.input_y: batch["y"],
                     self.keep_prob: keep_prob}

        # 运行会话
        _, summary, loss, predictions = sess.run([self.train_op, self.summary_op, 
                                                  self.loss, self.predictions],
                                                 feed_dict=feed_dict)
        
        return summary, loss, predictions

    def eval(self, sess, batch):
        """验证
        Args:
            sess: 会话
            batch: batch数据
        
        Return: 
            记录、损失和预测结果
        """
        feed_dict = {self.input_x: batch["x"],
                     self.input_y: batch["y"],
                     self.keep_prob: 1.0}                                               # 激活概率为1.0，使得dropout层失效

        summary, loss, predictions = sess.run([self.summary_op, self.loss, 
                                               self.predictions], 
                                              feed_dict=feed_dict)
        return summary, loss, predictions

    def infer(self, sess, inputs):
        """预测
        
        Args:
            sess: 会话
            inputs: batch数据
        
        Return: 
            预测结果
        """
        feed_dict = {self.input_x: inputs,
                     self.keep_prob: 1.0}

        predicts = sess.run(self.predictions, feed_dict=feed_dict)

        return predicts


