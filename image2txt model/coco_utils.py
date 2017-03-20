
"""Util functions for handling caption data"""

import os, json
import numpy as np
import h5py


def load_coco_data(base_dir='/home/ubuntu/COCO/dataset/COCO_captioning/',
                   max_train=None):
  data = {}
  
  # loading train&val captions, and train&val image index 
  caption_file = os.path.join(base_dir, 'coco2014_captions.h5')
  with h5py.File(caption_file, 'r') as f: # keys are: train_captions, val_captions, train_image_idxs, val_image_idxs
    for k, v in f.items():
      data[k] = np.asarray(v)

  train_feat_file = os.path.join(base_dir, 'train2014_v3_pool_3.npy')
  data['train_features'] = np.load(train_feat_file)

  val_feat_file = os.path.join(base_dir, 'val2014_v3_pool_3.npy')
  data['val_features'] = np.load(val_feat_file)

  dict_file = os.path.join(base_dir, 'coco2014_vocab.json')
  with open(dict_file, 'r') as f:
    dict_data = json.load(f)
    for k, v in dict_data.items():
      data[k] = v
  # convert string to int for the keys 
  data['idx_to_word'] = {int(k):v for k, v in data['idx_to_word'].items()}

  train_url_file = os.path.join(base_dir, 'train2014_urls.txt')
  with open(train_url_file, 'r') as f:
    train_urls = np.asarray([line.strip() for line in f])
  data['train_urls'] = train_urls

  val_url_file = os.path.join(base_dir, 'val2014_urls.txt')
  with open(val_url_file, 'r') as f:
    val_urls = np.asarray([line.strip() for line in f])
  data['val_urls'] = val_urls

  # Maybe subsample the training data
  if max_train is not None:
    num_train = data['train_captions'].shape[0]
    mask = np.random.randint(num_train, size=max_train)
    data['train_captions'] = data['train_captions'][mask]
    data['train_image_idx'] = data['train_image_idx'][mask]

  return data


def decode_captions(captions, idx_to_word):
  singleton = False
  if captions.ndim == 1:
    singleton = True
    captions = captions[None]
  decoded = []
  N, T = captions.shape
  for i in range(N):
    words = []
    for t in range(T):
      word = idx_to_word[captions[i, t]]
      if word != '<NULL>':
        words.append(word)
      if word == '<END>':
        break
    decoded.append(' '.join(words))
  if singleton:
    decoded = decoded[0]
  return decoded


def sample_coco_minibatch(data, batch_size=100, split='train'):
  split_size = data['%s_captions' % split].shape[0]
  mask = np.random.choice(split_size, batch_size)
  captions = data['%s_captions' % split][mask]
  image_idxs = data['%s_image_idx' % split][mask]
  image_features = data['%s_features' % split][image_idxs]
  urls = data['%s_urls' % split][image_idxs]
  return captions, image_features, urls

