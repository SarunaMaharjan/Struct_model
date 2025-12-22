#!/bin/bash

#chmod +x run_training.sh


python main2.py \
--train_file Hindi_data/train.txt \
--valid_file Hindi_data/valid.txt \
--cuda \
--batch_size 8 \
--epochs 1 \
--save struct_xlmr_final.pt \
--checkpoint_path checkpoint.pt \
--log_interval 200 \
--validation_interval 2500 \
--base_lr 1e-5 \
--parser_lr 1e-6 \
--resume checkpoint.pt \
--num_workers 24