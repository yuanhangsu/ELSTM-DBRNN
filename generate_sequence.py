from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import json
import sys
import random
import numpy as np
import time
from six.moves import xrange
import pdb
from shutil import copyfile

import tensorflow as tf

from utils import data_utils
from model import RNN_model
import configuration as Config

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("mode","train","")# train, inference
tf.flags.DEFINE_string("model","BASIC_RNN","BASIC_RNN or DBRNN or ENC_DEC_Att")
tf.flags.DEFINE_string("cell_mode","LSTM","LSTM, ELSTM")
tf.flags.DEFINE_string("no_previous","False","")
tf.flags.DEFINE_string("checkpoint_path", "./ckpt/",
					   "Model checkpoint file or directory containing a model checkpoint file.")
tf.flags.DEFINE_string("vocab_input_file", "./data/vocab_PTB_input.txt", "")
tf.flags.DEFINE_string("vocab_output_file", "./data/vocab_PTB_output.txt", "")
tf.flags.DEFINE_string("data_train_path","./data/data_PTB_train.json","data path for training")
tf.flags.DEFINE_string("data_val_path","./data/data_PTB_valid.json","data path for validation")
tf.flags.DEFINE_string("data_test_path","./data/data_PTB_test.json","data path for test")
tf.flags.DEFINE_integer("max_epochs",20,"maximum number of epochs for training")
tf.flags.DEFINE_integer("max_cell_length", 10,"maximum number of scaling variables")

data_mode = ''
if FLAGS.data_train_path[FLAGS.data_train_path.rfind('/'):].find('PTB') != -1:
	data_mode = 'PTB'

if FLAGS.no_previous == "True":
	no_previous = True
else:
	no_previous = False

def create_model(session, config, mode, forward_only, cell_mode=False):
	if FLAGS.model == 'DBRNN':
		if config.max_input_seq_length != config.max_output_seq_length:
			config.max_output_seq_length = config.max_input_seq_length
		model = RNN_model.DBRNNModel(config, mode, forward_only, 
			cell_mode=cell_mode, no_previous=no_previous, max_cell_length=FLAGS.max_cell_length)
	elif FLAGS.model == 'ENC_DEC_Att':
		model = RNN_model.ENC_DEC_ATTModel(config, mode, forward_only, True, 
			cell_mode=cell_mode, no_previous=no_previous, max_cell_length=FLAGS.max_cell_length)
	elif FLAGS.model == 'BASIC_RNN':
		if config.max_input_seq_length != config.max_output_seq_length:
			config.max_output_seq_length = config.max_input_seq_length
		model = RNN_model.BASIC_RNNModel(config, mode, forward_only, False, 
			cell_mode=cell_mode, no_previous=no_previous, max_cell_length=FLAGS.max_cell_length)
	else:
		NotImplementedError

	ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		# create checkpoint folder if not exist
		directory = os.path.dirname(FLAGS.checkpoint_path)
		if not os.path.exists(directory):
			os.makedirs(directory)
		print("Created %s %s model with fresh parameters."%(FLAGS.model,FLAGS.cell_mode))
		session.run(tf.compat.v1.global_variables_initializer())
	return model

def train():
	# data preparation	
	config = Config.ModelConfig()
	data_path, vocab_path = {},{}
	data_path['train'] = FLAGS.data_train_path
	data_path['val'] = FLAGS.data_val_path
	data_path['test'] = FLAGS.data_test_path
	vocab_path['input'] = FLAGS.vocab_input_file
	vocab_path['output'] = FLAGS.vocab_output_file
	#inputs = (data_path, vocab_path, data_mode, config.num_input_symbols, None, None)
	#inputs = (data_path, vocab_path, None, None, None)
	inputs = (data_path, vocab_path)
	(vocab_input,rev_vocab_input), (vocab_output,rev_vocab_output), data = \
		data_utils.prepare_data('PTB_data', inputs)
	data_train, data_val = data['train'], data['val']
	config.max_training_steps = int(config.steps_per_checkpoint*round(len(data_train)*float(FLAGS.max_epochs)
										/config.batch_size/config.steps_per_checkpoint))
	config.num_input_symbols = len(vocab_input)
	config.num_output_symbols = len(vocab_output)
	print("maximum training steps: " + str(config.max_training_steps))

	step_time, losses = 0.0, 0.0
	f_losses, b_losses = 0.0, 0.0
	current_step = 0
	previous_losses = []
	with tf.Session() as sess:
		model = create_model(sess, config, 'train', False, cell_mode=FLAGS.cell_mode)
		for i in xrange(config.max_training_steps):
			start_time = time.time()

			inputs, targets, target_weights = model.get_batch(data_train)
			b_loss, loss, f_loss = model.step(sess, inputs, targets, target_weights, False)

			# Once in a while, we save checkpoint, print statistics, and run evals.
			step_time += (time.time() - start_time) / config.steps_per_checkpoint
			losses += loss / config.steps_per_checkpoint
			if isinstance(f_loss, np.float32):
				f_losses += f_loss / config.steps_per_checkpoint
			if isinstance(b_loss, np.float32):
				b_losses += b_loss / config.steps_per_checkpoint
			current_step += 1
			if current_step % config.steps_per_checkpoint == 0:
				perplexity = math.exp(losses) if losses < 300 else float('inf')
				print ("global step %d learning rate %.4f step-time %.2f perplexity "
								"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
													step_time, perplexity))
				if f_losses:
					print (' '*20+"training forward perplexity %.2f" % (math.exp(f_losses) 
												if f_losses<300 else float('inf')))
				
				if b_losses:
					print (' '*20+"training backward perplexity %.2f" % (math.exp(b_losses) 
												if b_losses<300 else float('inf')))
				# Decrease learning rate if no improvement was seen over last 3 times.
				if len(previous_losses) > 2 and losses > max(previous_losses[-3:]):
					sess.run(model.learning_rate_decay_op)
				previous_losses.append(losses)
				# Save checkpoint and zero timer and loss.
				model.saver.save(sess, FLAGS.checkpoint_path, global_step=model.global_step)
				step_time, losses = 0.0, 0.0
				f_losses, b_losses = 0.0, 0.0
				# Run evals on development set and print their perplexity.
				val_inputs, val_targets, val_target_weights = model.get_batch(data_val)
				_, val_loss, val_loss_fb= model.step(sess, val_inputs, val_targets, val_target_weights, True)
				val_ppx = math.exp(val_loss) if val_loss < 300 else float('inf')
				print("  val: perplexity %.2f" % val_ppx)
				if isinstance(val_loss_fb, list):
					if len(val_loss_fb) == 2:
						f_val_loss, b_val_loss = val_loss_fb
					else:
						f_val_loss, b_val_loss = val_loss_fb, None
					if isinstance(f_val_loss,np.float32):
						print ("val: forward perplexity %.2f" % (math.exp(f_val_loss) 
														if f_val_loss<300 else float('inf')))
					if isinstance(b_val_loss,np.float32):
						print ("val: backward perplexity %.2f" % (math.exp(b_val_loss) 
														if b_val_loss<300 else float('inf')))
				sys.stdout.flush()

def test():
	# data preparation
	vocab_input, _ = data_utils.initialize_vocabulary(FLAGS.vocab_input_file)
	_, rev_vocab_output = data_utils.initialize_vocabulary(FLAGS.vocab_output_file)
	config = Config.ModelConfig()
	config.batch_size = 1 #only 1 batch for inferencing
	config.num_input_symbols = len(vocab_input)
	config.num_output_symbols = len(rev_vocab_output)
	# read in testing data if there is any
	if FLAGS.data_test_path:
		with open(FLAGS.data_test_path, 'rb') as f:
			data = json.load(f)
		f.close()
	with tf.Session() as sess:
		model = create_model(sess, config, 'inference', True, cell_mode=FLAGS.cell_mode)
		test_total_losses, count_sample = 0, 0
		for instance in data:
			inp_ids, tgt_ids = instance[0], instance[1]
			inputs, targets, target_weights = model.get_batch([[inp_ids, tgt_ids]])
			#DBRNN: output logits, brnn (forward then backward) output logits, tuple of [state, losses, fw_losses, bw_losses]
			#BASIC RNN: output logits, losses, state
			#ENC_DEC_ATT: output logits, losses, state
			output_logits, loss_or_brnn_logits, state = model.step(sess, inputs, targets, target_weights, True)

			if FLAGS.model == "DBRNN":
				_, test_losses, test_losses_fw, test_losses_bw = state
			else:
				test_losses = loss_or_brnn_logits
			#test losses per sample sentence
			test_total_losses += test_losses
			count_sample += 1
		
		print('PPX: '+str(math.exp(test_total_losses/float(count_sample))))
			
def main(_):
	if FLAGS.mode == 'train':
		train()
	elif FLAGS.mode == 'inference':
		test()

if __name__ == "__main__":
	tf.app.run()