#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

echo cleaning run directory on $gce

gcloud compute ssh $gce -- "cd $dir_rmt_run && rm -r * && mkdir -p bk"
gcloud compute ssh $gce -- "cd $dir_rmt_run && echo "" > submission_0_0-0.csv"
