#!/bin/bash

datename=$(date +%Y%m%d)-$(date +%H%M%S)
# touch $datename.log
nohup python -u kmeans.py > $datename.log
