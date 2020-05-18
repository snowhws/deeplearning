#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: wensong

import jieba
import sys

CLS_SAMPLE_NUM = 2000
MIN_TITLE_LEN = 5
MIN_CONTENT_LEN = 10

class_dict = {
    "auto.sohu.com/": 0,  # 汽车
    "business.sohu.com/": 1,  # 财经
    "it.sohu.com/": 2,  # IT
    "health.sohu.com/": 3,  # 健康
    "sports.sohu.com/": 4,  # 体育
    "travel.sohu.com/": 5,  # 旅游
    "learning.sohu.com/": 6,  # 教育
    "mil.news.sohu.com/": 7,  # 军事
    "cul.sohu.com/": 8,  # 文化
    "yule.sohu.com/": 9,  # 娱乐
    "women.sohu.com/": 10,  # 女性
}

statistic_dict = {}
for i in range(0, 11):
    statistic_dict[i] = 0


def formatting(sample):
    # filter by rules
    if "url" not in sample or "title" not in sample or "content" not in sample:
        return
    if len(sample["title"]) < MIN_TITLE_LEN or len(
            sample["content"]) < MIN_CONTENT_LEN:
        return
    # filter by url
    cls = 0
    for url in class_dict:
        if sample["url"].find(url) == -1:
            continue
        cls = class_dict[url]
        break
    if cls == 0:
        return
    # filter by CLS_SAMPLE_NUM
    if statistic_dict[cls] >= CLS_SAMPLE_NUM:
        return
    # title
    if sample["title"] == "":
        return
    t_splits = jieba.cut(sample["title"])
    # content
    if sample["content"] == "":
        return
    c_splits = jieba.cut(sample["content"])
    print str(cls) + "\t" + " ".join(t_splits) + "\t" + " ".join(c_splits)
    statistic_dict[cls] += 1


def read_news():
    sample = {}
    while 1:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.rstrip().strip()
        if line.find("</doc>") != -1:
            formatting(sample)
            sample = {}
        elif line.find("<url>") != -1:
            sample["url"] = line.replace("<url>", "").replace("</url>", "")
        elif line.find("<contenttitle>") != -1:
            sample["title"] = line.replace("<contenttitle>",
                                           "").replace("</contenttitle>", "")
        elif line.find("<content>") != -1:
            sample["content"] = line.replace("<content>",
                                             "").replace("</content>", "")
    formatting(sample)


read_news()
#print "========================"
#for cls in statistic_dict:
#    print str(cls) + "\tnum:" + str(statistic_dict[cls])
