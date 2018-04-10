#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

echo starting $gce

gcloud compute instances list
gcloud compute instances start $gce
gcloud compute instances list
