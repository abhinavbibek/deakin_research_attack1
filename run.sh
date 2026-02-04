#!/bin/bash
export CUDA_VISIBLE_DEVICES=6
python run_trigger_tiny.py
python run_attack_tiny.py

