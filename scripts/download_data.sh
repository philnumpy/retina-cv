#!/usr/bin/env bash
# Download the three datasets with the Kaggle API (needs ~/.kaggle credentials and,
# for APTOS, the competition rules accepted on the website). Safe to re-run.
set -u
cd "$(dirname "$0")/.."
RAW=data/raw
mkdir -p $RAW/drive $RAW/fives $RAW/aptos

retry() {  # the campus DNS drops out now and then, so retry a few times
  for i in 1 2 3 4 5 6 7 8; do "$@" && return 0; echo "retry $i: $*"; sleep 20; done; return 1
}

[ -n "$(ls $RAW/drive)" ] || { retry kaggle datasets download -d zionfuo/drive2004 -p $RAW/drive \
  && unzip -q -o $RAW/drive/*.zip -d $RAW/drive && rm $RAW/drive/*.zip; }
[ -n "$(ls $RAW/fives | grep -v -e rar -e log)" ] || { retry kaggle datasets download -d prashantdixit07/fives-a-fundus-image -p $RAW/fives \
  && unzip -q -o $RAW/fives/*.zip -d $RAW/fives && rm $RAW/fives/*.zip; }
[ -f $RAW/aptos/train.csv ] || { retry kaggle competitions download -c aptos2019-blindness-detection -p $RAW/aptos \
  && unzip -q -o $RAW/aptos/*.zip 'train.csv' 'train_images/*' -d $RAW/aptos && rm $RAW/aptos/*.zip; }
echo "download finished"; du -sh $RAW/*
