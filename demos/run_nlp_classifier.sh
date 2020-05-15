#!/usr/bin/env bash
# @Author wensong
# @Env tensorflow 1.5.0|Python2.7

run_textcnn() {
  python ./nlp_classifier.py \
    --task_name "TextCNN" \
    --max_seq_len 128
}

run_bilstm_att() {
  python ./nlp_classifier.py \
    --task_name "BILSTMAtt" \
    --max_seq_len 128
}

run_transformer() {
  python ./nlp_classifier.py \
    --task_name "Transformer" \
    --num_blocks 1 \
    --keep_prob 0.7 \
    --max_seq_len 128
}

#run_textcnn
#run_bilstm_att
run_transformer
