
"""Create word vectors initialization matrix using GloVe vectors"""

import json
import numpy as np 

TOTAL_VOCAB = 5004
EMBED_DIM = 300
INITIALIZER_SCALE = 0.08

dict_file = '/home/ubuntu/COCO/dataset/COCO_captioning/coco2014_vocab.json'
glove_file = '/home/ubuntu/COCO/GloVe/glove.42B.300d.txt'
save_glove_mat = '/home/ubuntu/COCO/dataset/COCO_captioning/glove_vocab'

glove_matrix = np.random.uniform(-INITIALIZER_SCALE, INITIALIZER_SCALE, (TOTAL_VOCAB, EMBED_DIM))

with open(dict_file, 'r') as f:
    dict_data = json.load(f)
for k, v in dict_data.items():
    data[k] = v
# convert string to int for the keys 
data['idx_to_word'] = {int(k):v for k, v in data['idx_to_word'].items()}
word_to_idx = data['word_to_idx']

total_word_replaced = 0
print_every = 100
with open(glove_file, 'r') as f:
    for line in f:
        line = line.strip()
        word = line.split(' ')[0]
        if word in word_to_idx:
            total_word_replaced += 1
            if total_word_replaced % print_every == 0:
                print(total_word_replaced)
            
            line = line.split(' ')[1:]
            word_vec = np.array([float(i) for i in line])
            
            glove_matrix[word_to_idx[word]] = word_vec
            
            if total_word_replaced == TOTAL_VOCAB - 4:
                break 

np.save(save_glove_mat, glove_matrix)











