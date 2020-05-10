#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../")
from utils.tf_utils import TFUtils
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
import time
import datetime
from models.nlp.classification.tf_bilstmatt_classifier import TFBILSTMATTClassifier
from models.nlp.classification.tf_textcnn_classifier import TFTextCNNClassifier


def init():
    '''初始化参数，使用tf的flags定义
    '''
    # 当前路径
    current_path = os.getcwd()
    # 样本集合相关参数
    tf.flags.DEFINE_float(
        "dev_sample_percentage", .1,
        "Percentage of the training data to use for validation")
    tf.flags.DEFINE_string(
        "positive_data_file",
        current_path + "../corpus/rt-polaritydata/rt-polarity.pos",
        "Data source for the positive data.")
    tf.flags.DEFINE_string(
        "negative_data_file",
        current_path + "../corpus/rt-polaritydata/rt-polarity.neg",
        "Data source for the negative data.")
    # 分类器公共参数
    tf.flags.DEFINE_integer("emb_size", 128,
                            "Dimensionality of word embedding (default: 128)")
    tf.flags.DEFINE_integer("max_seq_len", 1024,
                            "max len of input seq(default: 1024)")
    # TextCNN相关参数
    tf.flags.DEFINE_string("filter_sizes", "2,3,4,5",
                           "TextCNN filter sizes (default: '2,3,4,5')")
    tf.flags.DEFINE_integer(
        "num_filters", 128, "Number of filters per filter size (default: 128)")
    # 训练相关参数
    tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 100,
                            "Number of training epochs (default: 100)")
    tf.flags.DEFINE_float("keep_prob", 0.5,
                          "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 0.0,
                          "L2 regularization lambda (default: 0.0)")
    tf.flags.DEFINE_integer(
        "evaluate_every", 100,
        "Evaluate model on dev set after this many steps (default: 100)")
    tf.flags.DEFINE_integer("checkpoint_every", 100,
                            "Save model after this many steps (default: 100)")
    tf.flags.DEFINE_integer("num_checkpoints", 5,
                            "Number of checkpoints to store (default: 5)")
    #  设备及日志相关
    tf.flags.DEFINE_boolean(
        "allow_soft_placement", True,
        "Allow device soft device placement")  # 如果指定设备不存在，tf自动分配设备
    tf.flags.DEFINE_boolean("log_device_placement", False,
                            "Log placement of ops on devices")  # 是否打印备份日志


# 命令行参数
init()
FLAGS = tf.flags.FLAGS


def preprocess():
    # Data Preparation
    # ==================================================
    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file,
                                                  FLAGS.negative_data_file)

    # Build vocabulary
    # max_document_length = max([len(x.split(" ")) for x in x_text])
    max_document_length = FLAGS.sequence_lens
    vocab_processor = learn.preprocessing.VocabularyProcessor(
        max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[
        dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[
        dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev


def init_model(x_train,
               y_train,
               vocab_processor,
               x_dev,
               y_dev,
               model_name="lstm_att"):
    model = None
    if model_name == "cnn":
        model = TextCNN(seq_len=x_train.shape[1],
                        cls_num=y_train.shape[1],
                        voc_size=len(vocab_processor.vocabulary_),
                        emb_size=FLAGS.embedding_dim,
                        filter_sizes=list(
                            map(int, FLAGS.filter_sizes.split(","))),
                        num_filters=FLAGS.num_filters,
                        l2_reg_lambda=FLAGS.l2_reg_lambda)
    elif model_name == "lstm_att":
        config_dict = {
            "embedding_size": FLAGS.embedding_dim,
            "hidden_sizes": [128],
            "num_classes": 2,
            "sequence_length": 128,
            "classifier_type": "multi-class",
            "optimization": "adam",
            "max_grad_norm": 20,
            "learning_rate": 1e-1
        }
        model = BiLstmAttention(config_dict,
                                vocab_size=len(vocab_processor.vocabulary_),
                                word_vectors=None)
    return model


def train(x_train, y_train, vocab_processor, x_dev, y_dev):
    # Training
    # ==================================================
    # 实例化整个图作为默认图
    with tf.Graph().as_default():
        # 配置
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        # session
        sess = tf.Session(config=session_conf)
        # 默认会话
        with sess.as_default():
            # 创建分类器
            model = init_model(x_train, y_train, vocab_processor, x_dev, y_dev)
            # 全局训练步数
            global_step = tf.Variable(0, name="global_step", trainable=False)
            # 指定训练参数
            # 优化器
            optimizer = tf.train.AdamOptimizer(1e-3)
            # 计算loss
            grads_and_vars = optimizer.compute_gradients(model.loss)
            # 返回梯度更新的op
            train_op = optimizer.apply_gradients(grads_and_vars,
                                                 global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram(
                        "{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar(
                        "{}/grad/sparsity".format(v.name),
                        tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # 输出总结和模型output目录
            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(
                os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))

            # 方便在tensorBoard显示汇总信息
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", model.loss)
            acc_summary = tf.summary.scalar("accuracy", model.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge(
                [loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(
                train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir,
                                                       sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(
                os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            # 保存器
            saver = tf.train.Saver(tf.global_variables(),
                                   max_to_keep=FLAGS.num_checkpoints)

            # 保存词表
            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # 初始化
            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            # 单步训练
            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                # feed数据，对应占位符
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    model.keep_prob: FLAGS.dropout_keep_prob
                }
                # 运行会话，第一个参数fetches指定要获取的返回值，第二个参数feed输入词典
                _, step, summaries, loss, accuracy = sess.run(
                    fetches=[
                        train_op, global_step, train_summary_op, model.loss,
                        model.accuracy
                    ],
                    feed_dict=feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            # 评估
            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                    model.input_x: x_batch,
                    model.input_y: y_batch,
                    # 评估时候通过dropout设置不更新参数
                    model.keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, model.loss, model.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(
                    time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)

            # 生成batch：Generate batches
            # zip会将两个数组遍历打包成对元组
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)),
                                              FLAGS.batch_size,
                                              FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # 训练一次
                train_step(x_batch, y_batch)
                # 获取步数
                current_step = tf.train.global_step(sess, global_step)
                # 默认每100步评估一次
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                # 保存模型参数到目录checkpoint_prefix下
                if current_step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess,
                                      checkpoint_prefix,
                                      global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))


def main(argv=None):
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess()
    train(x_train, y_train, vocab_processor, x_dev, y_dev)


if __name__ == '__main__':
    '''运行
    '''
    # tf.app.run()
    tf.compat.v1.app.run()
