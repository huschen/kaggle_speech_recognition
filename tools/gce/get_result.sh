#!/bin/bash

set -e

sdir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source $sdir/config.sh

if [ -z "$1" ]; then
  msg0=''
else
  msg0=$1
fi

do_submit=0
do_tensorboard=0
echo getting results $msg0 from $gce
echo settings: submit=$do_submit, tensorboard=$do_tensorboard


tmp=`gcloud compute ssh $gce -- "cd $dir_rmt_run && ls -t submission*.csv | head -n1"`
csv=`echo $tmp |sed "s/$(printf '\r')\$//"`
echo $csv
date=`echo $csv | grep -o '[0-9]\+_[0-9]\+'`
echo date: $date
step=`echo $csv | grep -o '\-[0-9]*' | cut -f 2 -d '-'`
echo step: $step

if [ $date == '0_0' ]; then
  echo no available csv files!!
  exit 1
fi

new_dir=$dir_local_submissions/nn_"$date"-$step
mkdir -p $new_dir
cd $new_dir
echo getting results to $new_dir


gcloud compute scp $gce:$dir_rmt_run/training_$date.log training.log
msg1=`grep -o 'fold_idx=[0-9]*' training.log`
msg2=`grep "$step: \[validation\]" training.log | cut -f 6,9 -d " " `
echo submitting $msg0 $msg1, $msg2
kc="tensorflow-speech-recognition-challenge"

if [ $do_submit -eq 1 ]; then
  gcloud compute ssh $gce -- "kaggle competitions submit -f $dir_rmt_run/$csv -c $kc -m '$msg0 $msg1, $msg2'"
fi


echo '' > ztats.txt
gcloud compute scp $gce:$dir_rmt_run/$csv .
for kw in unknown silence yes no up down left right on off stop go; do
  count=`grep ,"$kw" $csv | wc -l`
  perc=`bc <<< "scale=2; $count*100/158538"`
  echo "$csv": "$kw=$count, $perc"
  echo "$csv": "$kw=$count, $perc" >> ztats.txt
done
gcloud compute scp $gce:$dir_rmt_run/submission_dbg_$date-$step.csv .
gcloud compute scp $gce:$dir_rmt_run/train*_$date-$step.csv .
python $sdir/../stats.py ../../dataset . >> ztats.txt


gcloud compute scp $gce:$dir_rmt_run/hist_*_$date.txt .
gcloud compute scp $gce:$dir_rmt_run/*_$date-$step.txt .
cat v_wrongs_$date-$step.txt | wc -l
sort -k1 -t, v_wrongs_$date-$step.txt > v_wrongs.txt
rm v_wrongs_$date-$step.txt
mv v_confusion_matrix_$date-$step.txt v_confusion_matrix.txt
for f in `ls hist_*_*.txt`; do
  short=`echo $f | cut -f 1,2 -d '_'`
  mv $f $short.txt
done


gcloud compute scp $gce:$dir_rmt_code/models.py .
gcloud compute scp $gce:$dir_rmt_code/`grep -o 'map_words.*.txt' training.log` .
gcloud compute scp $gce:$dir_rmt_code/`grep -o 'map_chars.*.txt' training.log | cut -f 1 -d "'"` .

gcloud compute ssh $gce -- "cd $dir_rmt_run && mv submission*$date-$step.csv bk/."

ckpts=ann_"$date".ckpt-$step
echo $ckpts
gcloud compute scp $gce:$dir_rmt_run/ckpts/"$ckpts"* .


if [ $do_tensorboard -eq 1 ]; then
  mkdir -p train
  log=`gcloud compute ssh $gce -- "cd $dir_rmt_run/logs/train/ && ls -t | head -n1"`
  log=`echo $log |sed "s/$(printf '\r')\$//"`
  gcloud compute scp $gce:$dir_rmt_run/logs/train/$log ./train/.

  mkdir -p validation
  log=`gcloud compute ssh $gce -- "cd $dir_rmt_run/logs/validation/ && ls -t | head -n1"`
  log=`echo $log |sed "s/$(printf '\r')\$//"`
  gcloud compute scp $gce:$dir_rmt_run/logs/validation/$log ./validation/.

  tensorboard --logdir=.
fi
