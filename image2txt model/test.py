

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

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

verbose = True
mode = 'inference'
directory = '/home/ubuntu/COCO/'

def _step_test(sess, data, batch_size, model, keep_prob):
    """
    Make a single gradient update for batch data. 
    """
    # Make a minibatch of training data
    minibatch = sample_coco_minibatch(data,
                  batch_size=batch_size,
                  split='val')
    captions, features, urls = minibatch

    
    # print out ground truth caption
    captions_in = captions[:, 0].reshape(-1, 1)
    
    state = None 
    final_preds = []
    current_pred = captions_in
    mask = np.zeros((batch_size, model_config.padded_length))
    mask[:, 0] = 1
    
    # get initial state using image feature 
    feed_dict = {model['image_feature']: features, 
                 model['keep_prob']: keep_prob}
    state = sess.run(model['initial_state'], feed_dict=feed_dict)
    
    # start to generate sentences
    for t in range(model_config.padded_length):
        feed_dict={model['input_seqs']: current_pred, 
                   model['initial_state']: state, 
                   model['input_mask']: mask, 
                   model['keep_prob']: keep_prob}
            
        current_pred, state = sess.run([model['preds'], model['final_state']], feed_dict=feed_dict)
        
        current_pred = current_pred.reshape(-1, 1)
        
        final_preds.append(current_pred)
        
    return final_preds, urls
    
# load data 
data = load_coco_data(base_dir = '/home/ubuntu/COCO/dataset/COCO_captioning/')

TOTAL_INFERENCE_STEP = 1
BATCH_SIZE_INFERENCE = 32

# Build the TensorFlow graph and train it
g = tf.Graph()
with g.as_default():
    # Build the model.
    model = build_model(model_config, mode, inference_batch = BATCH_SIZE_INFERENCE)
    
    # run training 
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
    
        sess.run(init)
    
        model['saver'].restore(sess, directory + "savedSession/model0.ckpt")
          
        print("Model restured! Last step run: ", sess.run(model['global_step']))
        
        for i in range(TOTAL_INFERENCE_STEP):
            captions_pred, urls = _step_test(sess, data, BATCH_SIZE_INFERENCE, model, 1.0) # the output is size (32, 16)
            captions_pred = [unpack.reshape(-1, 1) for unpack in captions_pred]
            captions_pred = np.concatenate(captions_pred, 1)
            
            captions_deco = decode_captions(captions_pred, data['idx_to_word'])
            
            for j in range(len(captions_deco)):
                img_name = directory + 'image_' + str(j) + '.jpg'
                img = image_from_url(urls[j])
                write_text_on_image(img, img_name, captions_deco[j])
            
           
        
  
  
  
  
  
  