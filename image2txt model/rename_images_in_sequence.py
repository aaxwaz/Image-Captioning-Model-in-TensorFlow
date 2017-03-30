import argparse
import os.path, os
import re
import sys
import tarfile

import numpy as np
import pandas as pd 
from six.moves import urllib
import tensorflow as tf
import re 

FLAGS = None

def main(_):
    image_dict = pd.read_csv(FLAGS.dict_dir) # cols: image_idx, image_id
    image_dict = image_dict.set_index('image_id')
    image_dict = image_dict['image_index'].to_dict()
    for img_name in os.listdir(FLAGS.image_dir):
        original_img_path = os.path.join(FLAGS.image_dir, img_name)
        temp_num = int(re.split('\.|_', img_name)[-2])
        temp_num = image_dict[temp_num] # convert image id to idx
        new_img_path = os.path.join(FLAGS.image_dir, '{0}.jpg'.format(temp_num))
        os.rename(original_img_path, new_img_path)
    print(".done")

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--dict_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/COCO_captioning/train_image_id_to_idx.csv',
      help="""\
      dir that contains train_image_id_to_idx.csv or val_image_id_to_idx.csv\
      """
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/train2014',
      help='Absolute path to directory containing images that are to be extracted.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)










