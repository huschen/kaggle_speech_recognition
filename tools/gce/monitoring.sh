#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

if [ $# -eq 0 ]; then
  download_validation=1
  echo downloading training and validation logs for tensorboard
else
  download_validation=0
  echo downloading training logs for tensorboard
fi


rm -rf $dir_local_run/log_dl
mkdir -p $dir_local_run/log_dl
cd $dir_local_run/log_dl

mkdir -p train
log=`gcloud compute ssh $gce -- "cd $dir_rmt_run/logs/train/ && ls -t | head -n1"`
log=`echo $log |sed "s/$(printf '\r')\$//"`
gcloud compute scp $gce:$dir_rmt_run/logs/train/$log ./train/.

if [ $download_validation -eq 1 ]; then
  mkdir -p validation
  log=`gcloud compute ssh $gce -- "cd $dir_rmt_run/logs/validation/ && ls -t | head -n1"`
  log=`echo $log |sed "s/$(printf '\r')\$//"`
  gcloud compute scp $gce:$dir_rmt_run/logs/validation/$log ./validation/.
fi

tensorboard --logdir=.
