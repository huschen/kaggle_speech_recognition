#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

$sdir/clean_gce.sh
echo stopping $gce

gcloud compute instances list
gcloud compute instances stop $gce
gcloud compute instances list
