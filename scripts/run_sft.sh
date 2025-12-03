#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
accelerate launch src/train/sft_train.py
