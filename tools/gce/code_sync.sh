#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

echo syncing code to $gce
cd $dir_local_code

for f in `ls *.txt *.py`; do
  gcloud compute scp $f $gce:$dir_rmt_code/.
done
