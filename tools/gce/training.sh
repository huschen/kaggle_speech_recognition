#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

if [  -z "$1"  ]; then
  settings=''
else
  settings="$1"
fi

$sdir/code_sync.sh

echo training with settings $settings
gcloud compute ssh $gce -- "cd $dir_rmt_code && python train.py $settings"
