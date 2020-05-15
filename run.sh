#!/usr/bin/env bash

# run jupyter
edit() {
   LANG=zn jupyter notebook
}

# 训练模型并自动收敛(收敛条件见参数)
train() {
  cd demos
  ./run_nlp_classifier.sh
  cd ..
}

# 分析模型和结果
analysis() {
  tensorboard --logdir=./log/
}

# edit
train
# analysis
