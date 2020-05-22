#!/usr/bin/env bash
# @Author wensong
# @Env tensorflow 1.5.0|Python2.7

run_textcnn() {
  python ./nlp_classifier.py \
    --task_name "TextCNN" 
}

run_bilstm_att() {
  python ./nlp_classifier.py \
    --keep_prob 0.5 \
    --attention_size 1 \
    --min_frequency 2 \
    --task_name "BILSTMAtt"
}

run_transformer() {
  python ./nlp_classifier.py \
    --task_name "Transformer" \
    --num_blocks 2 
}

run_han() {
  python ./nlp_classifier.py \
    --task_name "HierarchicalAtt" \
    --data_type "longtext" \
    --doc_separators "," \
    --min_frequency 2 \
    --max_doc_len 5 \
    --max_seq_len 15
}

run_longshort() {
  python ./nlp_classifier.py \
    --task_name "LongShortMixture" \
    --data_file "../corpus/nlp/chinese/news.samples" \
    --data_type "longtext_with_title" \
    --min_frequency 2 \
    --cls_num 11 \
    --max_doc_len 10 \
    --max_seq_len 20
}

#run_textcnn
#run_bilstm_att
#run_transformer
run_han
#run_longshort
