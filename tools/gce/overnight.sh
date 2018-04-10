#!/bin/bash

# gcloud compute ssh instance-x
# tmux

# running on gce
python train.py  --fold_idx 0
python train.py  --fold_idx 1
python train.py  --fold_idx 2
python train.py  --fold_idx 3
python train.py  --fold_idx 4
