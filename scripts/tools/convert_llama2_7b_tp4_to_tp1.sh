#!/bin/bash
PROJECT_DIR=$(pwd)

TP=1
TP4_DIR=$PROJECT_DIR/output/checkpoints/llama2-7b-tp4-mask-only-wikitext103-c4format/train_iters_800/ckpt
TP1_DIR=$PROJECT_DIR/output/checkpoints/llama2-7b-tp1-mask-only-wikitext103-c4format/train_iters_800/ckpt
TOKENIZER_MODEL=assets/checkpoints/llama2_7b_hf/tokenizer.model

OPTIONS=" \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --target-tensor-parallel-size ${TP} \
    --load-dir ${TP4_DIR} \
    --save-dir ${TP1_DIR} \
    --megatron-path ${PROJECT_DIR}"

echo $TP4_DIR
echo $TP1_DIR

pip install transformers wandb accelerate tqdm; cd $PROJECT_DIR/tools/checkpoint; python util.py $OPTIONS 