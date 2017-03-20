
"""Data preparation for training image captioning model 
This script will do the followings: 

1) Come up with a vocab list by pooling all training and val captions
2) Convert each word from captions to an integer based on the vocab list 
3) Produce image-name-index mapping, that maps an image to an integer based on its name (e.g. COCO_train2014_000000417432.jpg -> 1)
4) Rename all images using the image-name-index mapping above
"""

import json 
import os
import collections
import tensorflow as tf 
import re
import h5py
import argparse
import sys 
import numpy as np 
import pandas as pd

FLAGS = None
BUFFER_TOKENS = ['<NULL>', '<START>', '<END>', '<UNK>']

def _parse_sentence(s):
    s = s.replace('.', '')
    s = s.replace(',', '')
    s = s.replace('"', '')
    s = s.replace("'", '')
    s = s.lower()
    s = re.sub("\s\s+", " ", s)
    s = s.split(' ')
    return s

def preprocess_json_files(path_to_dir):
    """Extract captions from each file and combine into lists, as well as image ids, and returned as dict"""
    assert os.path.exists(path_to_dir), 'Path to directory of files does not exist!'
    results = {}
    for file in os.listdir(path_to_dir):
        if 'captions_train2014' not in file and 'captions_val2014' not in file: 
            print("Skipping file {}".format(file))
            continue
        temp_path = os.path.join(path_to_dir, file)
        with open(temp_path, 'r') as f:
            data = json.load(f)
        caps = data['annotations']
        images = [item['image_id'] for item in caps]
        urls = {}
        for img in data['images']:
            urls[img['id']] = img['flickr_url']
        caps = [_parse_sentence(item['caption']) for item in caps]
        results[file] = (caps, images, urls)
        del data
    # return dict of each file, having list of captions and image_ids
    """
    results is a dict of two files (train and val), each of which has a caps list (results[file1][0]) and a images list (results[file1][1]), and urls dict
    (results[file1][2]). cap list is a list of sentences(list of words), images list is a list of image ids(integers), and urls dict is a dict mapping each 
    image id to its url 
    """
    return results
    
def rename_images(dir, image_id_to_idx):
    image_dict = pd.read_csv(image_id_to_idx) # cols: image_idx, image_id
    image_dict = image_dict.set_index('image_id')
    image_dict = image_dict['image_index'].to_dict()
    for img_name in os.listdir(dir):
        original_img_path = os.path.join(dir, img_name)
        temp_num = int(re.split('\.|_', img_name)[-2])
        temp_num = image_dict[temp_num] # convert image id to idx
        new_img_path = os.path.join(dir, '{0}.jpg'.format(temp_num))
        os.rename(original_img_path, new_img_path)
    print("Renaming images for folder {} done. ".format(dir))

def main(_):

    ## get the vocaboluary 
    list_of_all_words = None 
    results = preprocess_json_files(FLAGS.file_dir)
    
    for k, v in results.items():
        if list_of_all_words is None:
            list_of_all_words = results[k][0].copy()
        else:
            list_of_all_words += results[k][0]
    list_of_all_words = [item for sublist in list_of_all_words for item in sublist]
    counter = collections.Counter(list_of_all_words)
    vocab = counter.most_common(FLAGS.total_vocab)
    print("\nVocab generated! Most, median and least frequent words from the vocab are: \n{0}\n{1}\n{2}\n".format(vocab[0], vocab[int(FLAGS.total_vocab/2)], vocab[-1]))
    
    ## create word_to_idx, and idx_to_word
    vocab = [i[0] for i in vocab]
    word_to_idx = {}
    idx_to_word = {}
    # add in BUFFER_TOKENS
    for i in range(len(BUFFER_TOKENS)):
        idx_to_word[int(i)] = BUFFER_TOKENS[i]
        word_to_idx[BUFFER_TOKENS[i]] = i

    for i in range(len(vocab)):
        word_to_idx[vocab[i]] = i + len(BUFFER_TOKENS)
        idx_to_word[int(i + len(BUFFER_TOKENS))] = vocab[i]
        
    word_dict = {}
    word_dict['idx_to_word'] = idx_to_word
    word_dict['word_to_idx'] = word_to_idx
    with open(os.path.join(FLAGS.file_dir, 'coco2014_vocab.json'), 'w') as f:
        json.dump(word_dict, f)
        
    ## convert sentences into encoding/integers
    # pad all sentence to length of FLAGS.padding_len - 2 
    def _convert_sentence_to_numbers(s):
        """Convert a sentence s (a list of words) to list of numbers using word_to_idx"""
        UNK_IDX = BUFFER_TOKENS.index('<UNK>')
        NULL_IDX = BUFFER_TOKENS.index('<NULL>')
        END_IDX = BUFFER_TOKENS.index('<END>')
        s_encoded = [word_to_idx.get(w, UNK_IDX) for w in s]
        s_encoded += [END_IDX]
        s_encoded += [NULL_IDX] * (FLAGS.padding_len - 1 - len(s_encoded))
        return s_encoded
    
    h = h5py.File(os.path.join(FLAGS.file_dir,'coco2014_captions.h5'), 'w')
    for k, _ in results.items():
        results_to_save = {}
        all_captions = results[k][0] # list of lists of words 
        all_images = results[k][1]
        all_urls = results[k][2]
        all_captions = [_convert_sentence_to_numbers(s) for s in all_captions] # list of numbers 
        valid_rows = [i for i in range(len(all_captions)) if len(all_captions[i]) == FLAGS.padding_len-1]
        all_captions= [row for row in all_captions if len(row) == FLAGS.padding_len-1]
        all_captions = np.array(all_captions)
        all_images = np.array(all_images)
        all_images = all_images[valid_rows]
        assert all_images.shape[0] == all_captions.shape[0], "Processing error! all_captions and all_images diff in length."
        # concatenate START and END tokens at two sides 
        START_TOKEN = BUFFER_TOKENS.index('<START>')
        #END_TOKEN = BUFFER_TOKENS.index('<END>')
        col_start = np.array([START_TOKEN] * all_images.shape[0]).reshape(-1, 1)
        #col_end = np.array([END_TOKEN] * all_images.shape[0]).reshape(-1, 1)
        all_captions = np.hstack([col_start, all_captions])
    
        ## create dicts that maps image ids to 0,...,total_images - image_idx_to_id, image_id_to_idx
        image_ids = set(all_images)
        image_idx = list(range(len(image_ids)))
        image_id_to_idx = {}
        image_idx_to_id = {}
        for idx, id in enumerate(image_ids):
            image_id_to_idx[id] = idx
            image_idx_to_id[idx] = id
        all_images_idx = np.array([image_id_to_idx.get(id) for id in all_images])
            
        ## save all the data 
        if 'train' in k:
            h.create_dataset('train_captions', data=all_captions)
            h.create_dataset('train_image_idx', data=all_images_idx)
            df = pd.DataFrame.from_dict(image_id_to_idx, 'index')
            df['image_id'] = df.index.values
            df.columns = ['image_index', 'image_id']
            df.to_csv(os.path.join(FLAGS.file_dir, 'train_image_id_to_idx.csv'), index = False)
            
            ## write urls file to local as train2014_urls.txt
            with open(os.path.join(FLAGS.file_dir, 'train2014_urls.txt'), 'w') as f:
                for idx in range(len(image_idx_to_id)):
                    this_url = all_urls[image_idx_to_id[idx]]
                    f.write(this_url + '\n')
            
        elif 'val' in k:
            h.create_dataset('val_captions', data=all_captions)
            h.create_dataset('val_image_idx', data=all_images_idx)
            df = pd.DataFrame.from_dict(image_id_to_idx, 'index')
            df['image_id'] = df.index.values
            df.columns = ['image_index', 'image_id']
            df.to_csv(os.path.join(FLAGS.file_dir, 'val_image_id_to_idx.csv'), index = False)
            
            ## write urls file to local as val2014_urls.txt
            with open(os.path.join(FLAGS.file_dir, 'val2014_urls.txt'), 'w') as f:
                for idx in range(len(image_idx_to_id)):
                    this_url = all_urls[image_idx_to_id[idx]]
                    f.write(this_url + '\n')
        else:
            print("Strange file name found in dir: {0}, \nit does not belong to train nor val, so it is not able to save results!".format(k))
    
    h.close()
    print("Data generation done.\n Start renaming images in sequence ...")

    if FLAGS.train_image_dir != '':
        train_dict = os.path.join(FLAGS.file_dir, 'train_image_id_to_idx.csv')
        rename_images(FLAGS.train_image_dir, train_dict)
        
    if FLAGS.val_image_dir != '':
        val_dict = os.path.join(FLAGS.file_dir, 'val_image_id_to_idx.csv')
        rename_images(FLAGS.val_image_dir, val_dict)
        
    print("all done. ")
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--file_dir',
      type=str,
      #default='C:\\Users\\WAWEIMIN\\Google Drive\\ShowAndTellWeimin\\coco_captioning\\original_captioning',
      default= '/home/ubuntu/COCO/dataset/COCO_captioning/', 
      help="""\
      Path to captions_train2014.json, captions_val2014.json\
      """
  )
  parser.add_argument(
      '--total_vocab',
      type=int,
      default=1000,
      help='Total number of vacobulary to use.'
  )
  parser.add_argument(
      '--padding_len',
      type=int,
      default=17,
      help='Total len of padding the sentence.'
  )
  parser.add_argument(
      '--train_image_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/train2014',
      help='Absolute path to training dir containing images that are to be renamed.'
  )
  parser.add_argument(
      '--val_image_dir',
      type=str,
      default='/home/ubuntu/COCO/dataset/val2014',
      help='Absolute path to val dir containing images that are to be renamed.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)





































