#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh
$sdir/clean_gce.sh

echo copying source code and overnight script to $gce

cd $dir_local_code
zip code.zip *.py *.txt
gcloud compute scp code.zip $gce:$dir_rmt_code/.
gcloud compute ssh $gce -- "cd $dir_rmt_code/ && unzip -o code.zip && rm code.zip"
rm -f code.zip

gcloud compute scp $sdir/overnight.sh $gce:$dir_rmt_code/.
