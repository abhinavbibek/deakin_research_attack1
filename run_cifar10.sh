#!/bin/bash

export CUDA_VISIBLE_DEVICES=3
python run_trigger_cifar10.py
python run_attack_cifar10.py
