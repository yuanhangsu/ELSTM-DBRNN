import json
import re
import random
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

PAD = b"_PAD"
BOS = b"_BOS"
EOS = b"_EOS"
UNK = b"_UNK"
START_VOCAB = [PAD, BOS, EOS, UNK]

PAD_ID = 0
BOS_ID = 1
EOS_ID = 2
UNK_ID = 3

DIGIT_RE = re.compile(br"\d")

def initialize_vocabulary(vocabulary_path):
  if gfile.Exists(vocabulary_path):
    rev_vocab = []
    with gfile.GFile(vocabulary_path, mode="rb") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [tf.compat.as_bytes(line.strip()) for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def read_data(data_path):
	if gfile.Exists(data_path):
		with open(data_path, 'r') as f:
			return json.load(f)
	else:
		raise ValueError("Data file %s not found.", data_path)

def to_ids(line, vocabulary, normalize_digits=True):
	line = tf.compat.as_bytes(line.rstrip('\n'))
	words = line.split()
	if not words: return []
	if not normalize_digits:
		ids = [vocabulary.get(w, UNK_ID) for w in words]
	else:
		ids = [vocabulary.get(DIGIT_RE.sub(b"0", w), UNK_ID) for w in words]
	return ids

def shuffle_data(data, portion):
	if np.sum([v for key,v in portion.iteritems()],dtype=np.float32) != 1.0:
		raise ValueError("Portion elements must sum to one, got %d instead", np.sum(portion))

	d_copy = data[:]
	random.shuffle(d_copy)

	data_shuffled = {}
	start = 0
	p = 0
	for key,v in portion.iteritems():
		size = int(len(d_copy)*v)
		if p == len(portion)-1:
			if start+size < len(d_copy):
				size = len(d_copy)-start
		data_shuffled[key] = d_copy[start:start+size]
		start += size
		p += 1

	return data_shuffled