
"""Extraction image features using pretrained Inception V3, and save as numpy arrays in local"""

import argparse
import os.path, os
import re
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf

FLAGS = None
pretrain_model_name = 'classify_image_graph_def.pb'
layer_to_extract = 'pool_3:0'
save_dir = '/home/ubuntu/COCO/dataset/train2014_v3_pool_3'

MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
#MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-v3-2016-03-01.tar.gz'

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.model_dir
  if not os.path.exists(dest_directory):
    os.makedirs(dest_directory)
  filename = MODEL_URL.split('/')[-1]
  filepath = os.path.join(dest_directory, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (
          filename, float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(MODEL_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
  tarfile.open(filepath, 'r:gz').extractall(dest_directory)

def create_graph():
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(os.path.join(
      FLAGS.model_dir, pretrain_model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')
    
def main(_):
  """Extract features for all images in image_dir.
  Args:
    FLAGS.image_dir: The directory where all images are stored.
    FLAGS.model_dir: The directory where model file is located. 
    FLAGS.save_dir:  File name of the final array 
    FLAGS.verbose:   Verbose frequency (0 for non-verbose)
  Returns:
    None
  """
  if not os.path.exists(FLAGS.image_dir):
      print("image_dir does not exit!")
      return None
      
  # download graph if not exists
  maybe_download_and_extract()
  
  # Creates graph from saved GraphDef.
  create_graph()
  
  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
    final_array = []
    extract_tensor = sess.graph.get_tensor_by_name(layer_to_extract)
    counter = 0
    print("There are total " + str(len(os.listdir(FLAGS.image_dir))) + " images to process.")
    for img_idx in range(len(os.listdir(FLAGS.image_dir))):
        if FLAGS.verbose > 0:
            counter += 1 
            if counter % FLAGS.verbose == 0:
                print("Processing images : {0}.jpg".format(img_idx))
            
        temp_path = os.path.join(FLAGS.image_dir, '{0}.jpg'.format(img_idx))
        
        image_data = tf.gfile.FastGFile(temp_path, 'rb').read()
        
        predictions = sess.run(extract_tensor, {'DecodeJpeg/contents:0': image_data})
        predictions = np.squeeze(predictions)

        final_array.append(predictions)

    final_array = np.array(final_array)

    np.save(FLAGS.save_dir, final_array)
    
    print("\n\ndone. Extracted features saved in: ", FLAGS.save_dir)
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  parser.add_argument(
      '--model_dir',
      type=str,
      default='/tmp/imagenet/',
      help="""\
      Path to classify_image_graph_def.pb\
      """
  )
  parser.add_argument(
      '--image_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/train2014/',
      help='Absolute path to directory containing images that are to be extracted.'
  )
  parser.add_argument(
      '--save_dir',
      type=str,
      default=save_dir,
      help='Absolute path where the final array will be saved.'
  )
  parser.add_argument(
      '--verbose',
      type=int,
      default=1000,
      help='Verbose of processing steps.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)














