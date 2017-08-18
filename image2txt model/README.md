## Steps to run

1: run prepare_captions.py to get the coco2014_captions.h5 which contains train_captions, train_image_idx, val_captions, val_image_idx, train and val image urls files, as well as two dict files - train_image_id_to_idx.csv, val_image_id_to_idx.csv

```shell
python prepare_captions.py --file_dir /home/ubuntu/COCO/dataset/COCO_captioning/ --total_vocab 2000 --padding_len 25
```

2: run rename_images_in_sequence.py to change all image names in both train2014 and val2014, using the two dict files generated in previous step. You only need to do this step once and for all!

```shell
python rename_images_in_sequence.py --dict_dir /home/ubuntu/COCO/dataset/COCO_captioning/train_image_id_to_idx.csv --image_dir /home/ubuntu/COCO/dataset/train2014
python rename_images_in_sequence.py --dict_dir /home/ubuntu/COCO/dataset/COCO_captioning/val_image_id_to_idx.csv --image_dir /home/ubuntu/COCO/dataset/val2014
```

3: extract features! run extract_features.py to extract inception v3 features for each of the train2014 and val2014 images in sequence. You only need to do this step once and for all!

```shell
python extract_features.py --model_dir /tmp/imagenet --image_dir /home/ubuntu/COCO/dataset/val2014 --save_dir /home/ubuntu/COCO/dataset/val2014_v3_pool_3 --verbose 500
python extract_features.py --model_dir /tmp/imagenet --image_dir /home/ubuntu/COCO/dataset/val2014 --save_dir /home/ubuntu/COCO/dataset/val2014_v3_pool_3 --verbose 500
```
