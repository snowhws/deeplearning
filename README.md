# Deeplearning手脚架

```wensong
这是一个快速搭建NLP、CV、推荐等DL方向的手脚架，包括基本的分类、回归、聚类、序列标注、双塔等常见模型的简洁实现。
```

![Python版本](https://img.shields.io/badge/Python-2.7|3.7-blue) ![Tensorflow版本](https://img.shields.io/badge/Tensorflow-1.5.0|2.1.0-blue) ![https://img.shields.io/badge/pytorch-1.3.1%7C1.4.0-blue](https://img.shields.io/badge/pytorch-1.3.1|1.4.0-blue) ![](https://img.shields.io/badge/MacOS-10.15.4-blue)



## 目录说明

```
.
|____layers
| |____tf_bilstm_att_layer.py
| |____tf_textcnn_layer.py
| |____tf_soft_att_layer.py
| |____tf_embedding_layer.py
| |____tf_classifier_layer.py
| |____tf_base_layer.py
|____LICENSE
|____corpus
| |____README.md
| |____nlp
| | |____english
| | | |____rt-polarity.neg
| | | |____rt-polarity.pos
| | |____chinese
|____utils
| |____tf_utils.pyc
| |______init__.py
| |____README.md
| |______init__.pyc
| |____tf_utils.py
|____models
| |____README.md
| |____nlp
| | |____classification
| | | |____tf_bilstmatt_classifier.py
| | | |____tf_textcnn_classifier.py
| | | |____tf_base_classifier.py
| | |____cluster
| | |____similarity
| | | |____README.md
| | |____seq2seq
| | |____sequence_labeling
| |____cv
| | |____README.md
|____run.sh
|____README.md
|____ci.sh
|____demos
| |______init__.py
| |____test.py
| |____README.md
| |____demo.py
```

## 说明文档

*   [模型文档](https://github.com/snowhws/deeplearning/tree/master/models)

## License

[Apache License 2.0](LICENSE)

