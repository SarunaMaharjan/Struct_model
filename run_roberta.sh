#!/bin/bash

python main_roberta.py \
--train_file Hindi_data/train.txt \
--valid_file Hindi_data/valid.txt \
--cuda \
--batch_size 8 \
--epochs 1 \
--save struct_roberta_final.pt \
--checkpoint_path roberta_checkpoint.pt \
--log_interval 200 \
--validation_interval 2500 \
--base_lr 1e-5 \
--parser_lr 1e-6 \
--model_path Roberta_hi \
--tokenizer_path hindi_RobertaTok \
--resume roberta_checkpoint.pt \
--num_workers 24
