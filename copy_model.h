

python python_utils/do_net_surgery.py \
  --out_net_def models/pascal_voc/VGG16/fast_rcnn_adv/train.prototxt \
  --net_surgery_json models/pascal_voc/VGG16/fast_rcnn_adv/init_weights2.json \
  --out_net_file output/fast_rcnn_adv/voc_2007_trainval/train_init.caffemodel
