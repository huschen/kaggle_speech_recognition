#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

gcloud compute instances list
gcloud compute instances start $gce
gcloud compute instances list


echo waiting for gce to be ready
gce_ready=0
for i in {1..6}; do
  if [ $gce_ready -eq 0 ]; then
    if ! gcloud compute ssh $gce -- "pwd" > /dev/null 2>&1; then
      echo ...; sleep 5
    else
      gce_ready=1
    fi
  fi
done

if [ $gce_ready -eq 0 ]; then
  echo gce is not ready, exiting...
  exit 1
fi


echo preparing source code and training data on $gce

cd $dir_local_code
zip code.zip *.py *.txt tools/prepare_data.py tools/bad_samples.txt
gcloud compute scp code.zip $gce:$dir_rmt_code/.
gcloud compute ssh $gce -- "cd $dir_rmt_code/ && unzip -o code.zip && rm code.zip"
rm -f code.zip

gcloud compute ssh $gce -- "cd $dir_rmt_code/dataset/train/ && rm -rf audio"
gcloud compute ssh $gce -- "$dir_rmt_code/tools/download_dataset.sh"

#gcloud compute ssh $gce -- "cd $dir_rmt_code && python train.py --num_batches 50"
