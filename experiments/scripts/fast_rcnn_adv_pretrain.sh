#!/bin/bash
# Usage:
# ./experiments/scripts/fast_rcnn_ohem.sh GPU NET DATASET [options args to {train,test}_net.py]
# DATASET is either pascal_voc or coco.
#
# Example:
# ./experiments/scripts/fast_rcnn_ohem.sh 0 VGG16 pascal_voc \
#   --set EXP_DIR foobar RNG_SEED 42 TRAIN.SCALES "[400, 500, 600, 700]"

set -x
set -e

export PYTHONUNBUFFERED="True"

GPU_ID=$1
NET=$2
NET_lc=${NET,,}
DATASET=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}
EXTRA_ARGS_SLUG=${EXTRA_ARGS// /_}

case $DATASET in
  pascal_voc)
    TRAIN_IMDB="voc_2007_trainval"
    TEST_IMDB="voc_2007_test"
    PT_DIR="pascal_voc"
    ITERS=25000
    ;;
  coco)
    echo "Support coming soon. Stay tuned!"
    exit
    # TRAIN_IMDB="coco_2014_train"
    # TEST_IMDB="coco_2014_minival"
    # PT_DIR="coco"
    # ITERS=280000
    ;;
  *)
    echo "No dataset given"
    exit
    ;;
esac

LOG="experiments/logs/fast_rcnn_adv_pretrain.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

# /nfs.yoda/xiaolonw/faster_rcnn/ohem/output/default/voc_2007_trainval/vgg16_fast_rcnn_iter_30000.caffemodel

time ./tools/train_net.py --gpu ${GPU_ID} \
  --solver models/${PT_DIR}/${NET}/fast_rcnn_adv_pretrain/solver.prototxt \
  --weights output/fast_rcnn_adv/voc_2007_trainval/fast_rcnn_std_iter_10000.caffemodel \
  --imdb ${TRAIN_IMDB} \
  --iters ${ITERS} \
  --cfg experiments/cfgs/fast_rcnn_adv_pretrain.yml \
  ${EXTRA_ARGS}



