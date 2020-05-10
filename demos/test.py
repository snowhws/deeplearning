#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import os
import sys
sys.path.append(os.getcwd() + "/../")
from utils.tf_utils import TFUtils


def run():
    tfile = open("../corpus/nlp/english/rt-polarity.pos")
    line = ""
    while True:
        line = tfile.readline()
        if not line:
            break
        print line
        print TFUtils.preprocess(line)
        print "================"


if __name__ == "__main__":
    run()
