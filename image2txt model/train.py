
"""Train the model"""

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
import os
import sys
import argparse

model_config = configuration.ModelConfig()
training_config = configuration.TrainingConfig()

FLAGS = None 
savedModelName = 'model1.0.ckpt'
mode = 'train'

def _run_validation(sess, data, batch_size, model, keep_prob):
    """
    Make a single gradient update for batch data. 
    """
    # Make a minibatch of training data
    minibatch = sample_coco_minibatch(data,
                  batch_size=batch_size,
                  split='val')
    captions, features, urls = minibatch
    
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

def _step(sess, data, train_op, model, keep_prob):
    """
    Make a single gradient update for batch data. 
    """
    # Make a minibatch of training data
    minibatch = sample_coco_minibatch(data,
                  batch_size=model_config.batch_size,
                  split='train')
    captions, features, urls = minibatch

    captions_in = captions[:, :-1]
    captions_out = captions[:, 1:]

    mask = (captions_out != model_config._null)

    _, total_loss_value= sess.run([train_op, model['total_loss']], 
                                  feed_dict={model['image_feature']: features, 
                                             model['input_seqs']: captions_in, 
                                             model['target_seqs']: captions_out, 
                                             model['input_mask']: mask, 
                                             model['keep_prob']: keep_prob})

    return total_loss_value
    
def main(_):
    # load data 
    data = load_coco_data(FLAGS.data_dir)
    
    # force padded_length equal to padded_length - 1
    # model_config.padded_length = len(data['train_captions'][0]) - 1

    tf.reset_default_graph()
    
    # Build the TensorFlow graph and train it
    g = tf.Graph()
    with g.as_default():

        # Build the model. If FLAGS.glove_vocab is null, we do not initialize the model with word vectors; if not, we initialize with glove vectors
        if FLAGS.glove_vocab is '':               
            model = build_model(model_config, mode=mode)
        else:
            glove_vocab = np.load(FLAGS.glove_vocab)
            model = build_model(model_config, mode=mode, glove_vocab=glove_vocab)

        # Set up the learning rate.
        learning_rate_decay_fn = None
        learning_rate = tf.constant(training_config.initial_learning_rate)
        if training_config.learning_rate_decay_factor > 0:
            num_batches_per_epoch = (training_config.num_examples_per_epoch / model_config.batch_size)
            decay_steps = int(num_batches_per_epoch *
                              training_config.num_epochs_per_decay)

            def _learning_rate_decay_fn(learning_rate, global_step):
              return tf.train.exponential_decay(
                  learning_rate,
                  global_step,
                  decay_steps=decay_steps,
                  decay_rate=training_config.learning_rate_decay_factor,
                  staircase=True)

            learning_rate_decay_fn = _learning_rate_decay_fn

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model['total_loss'],
            global_step=model['global_step'],
            learning_rate=learning_rate,
            optimizer=training_config.optimizer,
            clip_gradients=training_config.clip_gradients,
            learning_rate_decay_fn=learning_rate_decay_fn)

        # initialize all variables 
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            
            num_epochs = training_config.total_num_epochs

            num_train = data['train_captions'].shape[0]
            iterations_per_epoch = max(num_train / model_config.batch_size, 1)
            num_iterations = int(num_epochs * iterations_per_epoch)
            
            # Set up some variables for book-keeping
            epoch = 0
            best_val_acc = 0
            best_params = {}
            loss_history = []
            train_acc_history = []
            val_acc_history = []

            print("\n\nTotal training iter: ", num_iterations, "\n\n")
            time_now = datetime.now()
            for t in range(num_iterations):
            
                total_loss_value = _step(sess, data, train_op, model, model_config.lstm_dropout_keep_prob) # run each training step 
                
                loss_history.append(total_loss_value)

                # Print out training loss
                if FLAGS.print_every > 0 and t % FLAGS.print_every == 0:
                    print('(Iteration %d / %d) loss: %f, and time eclipsed: %.2f minutes' % (
                        t + 1, num_iterations, float(loss_history[-1]), (datetime.now() - time_now).seconds/60.0))
                        
                # Print out some image sample results 
                if FLAGS.sample_every > 0 and (t+1) % FLAGS.sample_every == 0:
                    temp_dir = os.path.join(FLAGS.sample_dir, 'temp_dir_{}//'.format(t+1))
                    if not os.path.exists(temp_dir):
                        os.makedirs(temp_dir)
                    captions_pred, urls = _run_validation(sess, data, model_config.batch_size, model, 1.0) # the output is size (32, 16)
                    captions_pred = [unpack.reshape(-1, 1) for unpack in captions_pred]
                    captions_pred = np.concatenate(captions_pred, 1)
                    
                    captions_deco = decode_captions(captions_pred, data['idx_to_word'])
                    
                    for j in range(len(captions_deco)):
                        img_name = os.path.join(temp_dir, 'image_{}.jpg'.format(j))
                        img = image_from_url(urls[j])
                        write_text_on_image(img, img_name, captions_deco[j])
                
                # save the model continuously to avoid interruption 
                if FLAGS.saveModel_every > 0 and (t+1) % FLAGS.saveModel_every == 0:
                    if not os.path.exists(FLAGS.savedSession_dir):
                        os.makedirs(FLAGS.savedSession_dir)
                    checkpoint_name = savedModelName[:-5] + '_checkpoint{}.ckpt'.format(t+1)
                    save_path = model['saver'].save(sess, os.path.join(FLAGS.savedSession_dir, checkpoint_name))
                        
            if not os.path.exists(FLAGS.savedSession_dir):
                os.makedirs(FLAGS.savedSession_dir)
            save_path = model['saver'].save(sess, os.path.join(FLAGS.savedSession_dir, savedModelName))
            print("done. Model saved at: ", os.path.join(FLAGS.savedSession_dir, savedModelName))  
            
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  # classify_image_graph_def.pb:
  #   Binary representation of the GraphDef protocol buffer.
  parser.add_argument(
      '--savedSession_dir',
      type=str,
      default='/home/ubuntu/COCO/savedSession/',
      help="""\
      Directory where your created model / session will be saved.\
      """
  )
  parser.add_argument(
      '--data_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/COCO_captioning/',
      help='Directory where all your training and validation data can be found.'
  )
  parser.add_argument(
      '--glove_vocab',
      type=str,
      default='',
      help='Directory to glove vocab matrix - glove_vocab.npy - for initialization. Null for not using it. '
  )
  parser.add_argument(
      '--sample_dir',
      type=str,
      default='/home/ubuntu/COCO/progress_sample/',
      help='Directory where all intermediate samples will be saved.'
  )
  parser.add_argument(
      '--print_every',
      type=int,
      default=50,
      help='Num of steps to print your training loss. 0 for not printing/'
  )
  parser.add_argument(
      '--sample_every',
      type=int,
      default=5000,
      help='Num of steps to generate captions on some validation images. 0 for not validating.'
  )
  parser.add_argument(
      '--saveModel_every',
      type=int,
      default=5000,
      help='Num of steps to save model checkpoint. 0 for not doing so.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

