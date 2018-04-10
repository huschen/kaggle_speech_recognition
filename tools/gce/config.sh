#!/bin/bash

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
dir_local_code="$sdir/../../"
dir_local_run="$dir_local_code/run"
dir_local_submissions="$dir_local_code/submissions"

gce=instance-3
dir_project="~/kaggle_speech_recognition/"

dir_rmt_code="$dir_project"
dir_rmt_run="$dir_project/run/"
