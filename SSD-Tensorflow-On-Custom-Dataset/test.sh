# -*- coding: UTF-8 -*-
DATASET_DIR=./tfrecords/test/   
EVAL_DIR=./log_eval/    
CHECKPOINT_PATH=./log/model.ckpt-40000  
 python3 ./eval_ssd_network.py \
        --eval_dir=${EVAL_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --dataset_name=pascalvoc_2007 \
        --dataset_split_name=test \
        --model_name=ssd_300_vgg \
        --checkpoint_path=${CHECKPOINT_PATH} \
        --batch_size=1