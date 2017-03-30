
"""Predict captions on test images using trained model, with beam search method"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datetime import datetime 
import configuration
from ShowAndTellModel import build_model
from coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from image_utils import image_from_url, write_text_on_image
import numpy as np
import scipy.misc
from scipy.misc import imread
import pandas as pd
import os
from six.moves import urllib
import sys 
import tarfile
import json
import argparse
from caption_generator import * 

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None
verbose = True
mode = 'inference'

pretrain_model_name = 'classify_image_graph_def.pb'
layer_to_extract = 'pool_3:0'
MODEL_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

def maybe_download_and_extract():
  """Download and extract model tar file."""
  dest_directory = FLAGS.pretrain_dir
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
      FLAGS.pretrain_dir, pretrain_model_name), 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

def extract_features(image_dir):

    if not os.path.exists(image_dir):
        print("image_dir does not exit!")
        return None

    maybe_download_and_extract()
    
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
        print("There are total " + str(len(os.listdir(image_dir))) + " images to process.")
        all_image_names = os.listdir(image_dir)
        all_image_names = pd.DataFrame({'file_name':all_image_names})
        
        for img in all_image_names['file_name'].values:
                
            temp_path = os.path.join(image_dir, img)
            
            image_data = tf.gfile.FastGFile(temp_path, 'rb').read()
            
            predictions = sess.run(extract_tensor, {'DecodeJpeg/contents:0': image_data})
            predictions = np.squeeze(predictions)

            final_array.append(predictions)

        final_array = np.array(final_array)
    return final_array, all_image_names


def run_inference(sess, features, generator, keep_prob):

    batch_size = features.shape[0]

    final_preds = []

    for i in range(batch_size):
        feature = features[i].reshape(1, -1)
        pred = generator.beam_search(sess, feature)
        pred = pred[0].sentence
        final_preds.append(np.array(pred))
        
    return final_preds

def main(_):
    
    # load dictionary 
    data = {}
    with open(FLAGS.dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v
    data['idx_to_word'] = {int(k):v for k, v in data['idx_to_word'].items()}

    # extract all features 
    features, all_image_names = extract_features(FLAGS.test_dir)
    
    # Build the TensorFlow graph and train it
    g = tf.Graph()
    with g.as_default():
        num_of_images = len(os.listdir(FLAGS.test_dir))
        print("Inferencing on {} images".format(num_of_images))
        
        # Build the model.
        model = build_model(model_config, mode, inference_batch = 1)
        
        # Initialize beam search Caption Generator 
        generator = CaptionGenerator(model, data['word_to_idx'], max_caption_length = model_config.padded_length-1)
        
        # run training 
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
        
            sess.run(init)
        
            model['saver'].restore(sess, FLAGS.saved_sess)
              
            print("Model restored! Last step run: ", sess.run(model['global_step']))
            
            # predictions 
            final_preds = run_inference(sess, features, generator, 1.0)
            captions_pred = [unpack.reshape(-1, 1) for unpack in final_preds]
            #captions_pred = np.concatenate(captions_pred, 1)
            captions_deco= []
            for cap in captions_pred:
                dec = decode_captions(cap.reshape(-1, 1), data['idx_to_word'])
                dec = ' '.join(dec)
                captions_deco.append(dec)
            
            # saved the images with captions written on them
            if not os.path.exists(FLAGS.results_dir):
                os.makedirs(FLAGS.results_dir)
            for j in range(len(captions_deco)):
                this_image_name = all_image_names['file_name'].values[j]
                img_name = os.path.join(FLAGS.results_dir, this_image_name)
                img = imread(os.path.join(FLAGS.test_dir, this_image_name))
                write_text_on_image(img, img_name, captions_deco[j])
    print("\ndone.")
               
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--pretrain_dir',
      type=str,
      default= '/tmp/imagenet/',
      help="""\
      Path to pretrained model (if not found, will download from web)\
      """
  )
  parser.add_argument(
      '--test_dir',
      type=str,
      default= '/home/ubuntu/COCO/testImages/', 
      help="""\
      Path to dir of test images to be predicted\
      """
  )
  parser.add_argument(
      '--results_dir',
      type=str,
      default= '/home/ubuntu/COCO/savedTestImages/', 
      help="""\
      Path to dir of predicted test images\
      """
  )
  parser.add_argument(
      '--saved_sess',
      type=str,
      default= "/home/ubuntu/COCO/savedSession/model0.ckpt", 
      help="""\
      Path to saved session\
      """
  )
  parser.add_argument(
      '--dict_file',
      type=str,
      default= '/home/ubuntu/COCO/dataset/COCO_captioning/coco2014_vocab.json', 
      help="""\
      Path to dictionary file\
      """
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

    
  
  
  
  
  
  