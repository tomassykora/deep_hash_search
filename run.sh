#!/bin/bash
module add tensorflow-1.3.0-gpu-python3
pip install Keras --user
python3 /storage/brno6/home/cepin/deep_hash_search/resnet.py > /storage/brno6/home/cepin/deep_hash_search/out_new.log 
