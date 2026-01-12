#!/bin/bash
# Tiny-ImageNet
export CUDA_VISIBLE_DEVICES=0
python run_trigger_tiny.py
python run_attack_tiny.py

