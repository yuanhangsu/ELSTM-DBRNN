import json
import re
import random
import collections
import numpy as np
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.platform import gfile

from . common_utils import *

def _read_words(filename):
  with tf.gfile.GFile(filename, "r") as f:
    return f.read().decode("utf-8").replace("\n", "<eos>").split()

def _build_vocab(filename):
	data = _read_words(filename)

	counter = collections.Counter(data)
	count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

	words, _ = list(zip(*count_pairs))
	words = START_VOCAB + list(words)
	word_to_id = dict(zip(words, range(len(words))))

	return word_to_id, words

def _file_to_line_ids(filename, word_to_id):
	all_ids = []
	with tf.gfile.GFile(filename, "r") as f:
		for line in f.readlines():
			ids = to_ids(line, word_to_id, 
						normalize_digits=True)
			if ids: all_ids.append(ids)
	return all_ids

def PTB_prepare_data(data_path, vocab_path):
	train_path, train_json_path = data_path['train'], data_path['train'].replace('txt','json')
	valid_path, valid_json_path = data_path['val'], data_path['val'].replace('txt','json')
	test_path, test_json_path = data_path['test'], data_path['test'].replace('txt','json')


	data_ids = {}
	if gfile.Exists(train_json_path) and \
		gfile.Exists(valid_json_path) and \
		gfile.Exists(test_json_path):
		data_ids['train'] = read_data(train_json_path)
		data_ids['val'] = read_data(valid_json_path)
		data_ids['test'] = read_data(test_json_path)

		vocab_input, rev_vocab_input = initialize_vocabulary(vocab_path['input'])
		vocab_output, rev_vocab_output = initialize_vocabulary(vocab_path['output'])

		return (vocab_input, rev_vocab_input), (vocab_output, rev_vocab_output), data_ids

	word_to_id, id_to_word = _build_vocab(train_path)
	# create vocabulary file
	with open(vocab_path, 'wb') as f:
		for word in id_to_word:
			f.write(word+b"\n")

	# sentences to ids
	data_ids['train'] = _file_to_line_ids(train_path, word_to_id)
	data_ids['val'] = _file_to_line_ids(valid_path, word_to_id)
	data_ids['test'] = _file_to_line_ids(test_path, word_to_id)

	# make input, target pairs
	data_ids['train'] = [ [line[:-1], line[1:]] for line in data_ids['train']]
	data_ids['val'] = [ [line[:-1], line[1:]] for line in data_ids['val']]
	data_ids['test'] = [ [line[:-1], line[1:]] for line in data_ids['test']]

	# shuffle the data
	data_ids['train'] = shuffle_data(data_ids['train'], {'train':1.0})['train']
	data_ids['val'] = shuffle_data(data_ids['val'], {'val':1.0})['val']
	data_ids['test'] = shuffle_data(data_ids['test'], {'test':1.0})['test']

	# dump data to json files
	if gfile.Exists(train_json_path): gfile.Remove(train_json_path)
	if gfile.Exists(valid_json_path): gfile.Remove(valid_json_path)
	if gfile.Exists(test_json_path): gfile.Remove(test_json_path)

	with open(train_json_path, 'wb') as f:
		json.dump(data_ids['train'], f, sort_keys=True, indent=4)
	with open(valid_json_path, 'wb') as f:
		json.dump(data_ids['val'], f, sort_keys=True, indent=4)
	with open(test_json_path, 'wb') as f:
		json.dump(data_ids['test'], f, sort_keys=True, indent=4)

	return (word_to_id, id_to_word), (word_to_id, id_to_word), data_ids

def main():
	data_path = {'train': '../data/ptb/ptb.train.txt', 
					'val': '../data/ptb/ptb.valid.txt', 
					'test': '../data/ptb/ptb.test.txt'}
	PTB_prepare_data(data_path, '../data/PTB_vocab.txt')

if __name__ == "__main__":
	main()
