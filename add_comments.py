#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong
import sys


def add_space(line):
    nline = line
    for i in range(0, 60 - len(line)):
        nline += " "
    return nline


while True:
    line = sys.stdin.readline()
    if not line:
        break
    line = line.strip()

    if line.find(" corpus") != -1:
        print add_space(line) + "# 语料"
    elif line.find(" demos") != -1:
        print add_space(line) + "# 执行"
    elif line.find(" executes") != -1:
        print add_space(line) + "# 执行内核"
    elif line.find(" executor") != -1:
        print add_space(line) + "# 执行器"
    elif line.find(" graph_processor") != -1:
        print add_space(line) + "# 构图处理器"
    elif line.find(" init_processor") != -1:
        print add_space(line) + "# 参数处理器"
    elif line.find(" pre_processor") != -1:
        print add_space(line) + "# 语料处理器"
    elif line.find(" session_processor") != -1:
        print add_space(line) + "# 会话处理器"
    elif line.find(" nlp_classifier.py") != -1:
        print add_space(line) + "# nlp分类器"
    elif line.find(" layers") != -1:
        print add_space(line) + "# 高级层"
    elif line.find(" models") != -1:
        print add_space(line) + "# 模型层"
    elif line.find(" tf_utils.py") != -1:
        print add_space(line) + "# 工具类"
    elif line.find(" run.sh") != -1:
        print add_space(line) + "# 执行入口"
    elif line.find(" save_models") != -1:
        print add_space(line) + "# 模型保存地址"
    else:
        print line
