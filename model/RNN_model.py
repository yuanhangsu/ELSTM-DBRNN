import sys
import random
import numpy as np
from six.moves import xrange

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs

sys.path.insert(0,'..')
from utils import data_utils
import model.model_utils as model_utils
import model.RNN_cell as RNN_cell


class RNNModel(object):
	# keep stats now only works for ELSTMS, and only set for BASIC_RNN model
	def __init__(self, config, mode, cell_mode='ELSTM', no_previous=False, max_cell_length=0):
		assert mode in ["train", "inference"]
		self.batch_size = config.batch_size
		self.cell_units = config.cell_units
		self.cell_mul = config.cell_mul
		self.not_shared = 0
		self.num_input_symbols = config.num_input_symbols
		self.num_output_symbols = config.num_output_symbols
		self.num_layers = config.num_layers
		self.max_input_seq_length = config.max_input_seq_length
		self.max_output_seq_length = config.max_output_seq_length
		self.embedding_size = config.embedding_size
		self.act_func = config.activation_func
		self.num_samples = config.num_samples
		self.max_gradient_norm = config.max_gradient_norm
		self.no_previous = no_previous
		self.max_checkpoints_to_keep = config.max_checkpoints_to_keep
		self.mode = mode
		self.inputs = []
		self.targets = []
		self.weights = []
		self.initial_state = None


		self.global_step = tf.Variable(0, name="GlobalStep", trainable=False)
		self.learning_rate = tf.Variable(float(config.learning_rate), name="LearningRate", trainable=False)
		self.learning_rate_decay_op = self.learning_rate.assign(
			self.learning_rate * config.learning_rate_decay_factor)

		# data feed
		for i in xrange(self.max_input_seq_length):
			self.inputs.append(tf.compat.v1.placeholder(dtype=tf.int64,
	                                  shape=[None],  # batch_size
	                                  name="input_feed{0}".format(i)))
		for i in xrange(self.max_output_seq_length):
			self.targets.append(tf.compat.v1.placeholder(dtype=tf.int64,
	                                  shape=[None],  # batch_size
	                                  name="target_feed{0}".format(i)))
			self.weights.append(tf.compat.v1.placeholder(dtype=tf.float32,
	                                  shape=[None],  # batch_size
	                                  name="weights{0}".format(i)))
		
  		# cell definition
		if cell_mode == 'LSTM':
			single_cell = RNN_cell.LSTMCell(self.cell_units, activation=self.act_func, state_is_tuple=False)
		elif cell_mode == 'ELSTM':
			single_cell = RNN_cell.ELSTMCell(self.cell_units, state_is_tuple=False, 
				no_previous=self.no_previous, activation=self.act_func)
			self.not_shared = max_cell_length
			print(max_cell_length)
		else:
			raise NotImplementedError

		if self.num_layers > 1:
			self.cell = tf.nn.rnn_cell.MultiRNNCell([single_cell] * self.num_layers)
		else:
			self.cell = single_cell
		self.zero_state = self.cell.zero_state(self.batch_size, tf.float32)
		try:
			state_size = self.zero_state.get_shape()
		except:
			state_size = self.zero_state[0].get_shape()
		self.initial_state = tf.compat.v1.placeholder(dtype=tf.float32,
									shape=state_size,
									name="initial_state")

	def get_batch(self, data):
		batch_inputs, batch_targets = [],[]
		for batch_idx in xrange(self.batch_size):
			instance = random.choice(data)
			# instance is input, target pairs
			input_ids, output_ids = instance
			# reverse input and target sequences
			#input_ids = list(reversed(input_ids))
			#output_ids = list(reversed(output_ids))
			# end of reverse
			input_ids = input_ids[:self.max_input_seq_length]
			output_ids += [data_utils.EOS_ID]
			if len(output_ids) > self.max_output_seq_length:
				output_ids = output_ids[:self.max_output_seq_length-1]+[data_utils.EOS_ID]
			
			# create batch
			pad = [data_utils.PAD_ID] * (len(self.inputs)-len(input_ids))
			batch_inputs.append(input_ids+pad)
			#if self.mode == 'train':
			batch_targets.append(output_ids+
					[data_utils.PAD_ID] * (len(self.targets)-len(output_ids)))

		#reindex batch inputs into sequence inputs
		inputs, targets, target_weights = [],[],[]
		for length_idx in xrange(len(self.inputs)):
			inputs.append(np.array([batch_inputs[batch_idx][length_idx]
								for batch_idx in xrange(self.batch_size)], dtype=np.int32))
		#if self.mode == 'train':
		if batch_targets:
			for length_idx in xrange(len(self.targets)):
				targets.append(np.array([batch_targets[batch_idx][length_idx]
					for batch_idx in xrange(self.batch_size)], dtype=np.int32))
				# We set weight to 0 if the corresponding target is a PAD symbol.
				target_weight = np.ones(self.batch_size, dtype=np.float32)
				for batch_idx in xrange(self.batch_size):
					if batch_targets[batch_idx][length_idx] == data_utils.PAD_ID:
						target_weight[batch_idx] = 0.0
				target_weights.append(target_weight)

		return inputs, targets, target_weights

class BASIC_RNNModel(RNNModel):
	def __init__(self, config, mode, forward_only, feed_previous, cell_mode=None, no_previous=False, max_cell_length=None):
		super(BASIC_RNNModel, self).__init__(config, mode, cell_mode=cell_mode, no_previous=no_previous, max_cell_length=max_cell_length)

		# make sampled softmax
		output_projection = None
		softmax_loss_function = None
		if self.num_samples and self.num_samples < self.num_output_symbols:
			w = tf.get_variable("proj_w", [self.cell_units, self.num_output_symbols])
			w_t = tf.transpose(w)
			b = tf.get_variable("proj_b", [self.num_output_symbols])
			output_projection = (w, b)
			def sampled_loss(labels, inputs):
				labels = tf.reshape(labels, [-1, 1])
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b, tf.float32)
				local_inputs = tf.cast(inputs, tf.float32)
				return tf.nn.sampled_softmax_loss(
					weights=local_w_t,
					biases=local_b,
					labels=labels,
					inputs=local_inputs,
					num_sampled=self.num_samples,
					num_classes=self.num_output_symbols)
			softmax_loss_function = sampled_loss

		# one to one learning task
		def Net(inputs,feed_previous,initial_state=None):
			return model_utils.basic_rnn(
				inputs, self.cell,
				num_input_symbols=self.num_input_symbols,
				num_output_symbols=self.num_output_symbols,
				embedding_size=self.embedding_size,
				output_projection=output_projection,
				feed_previous=feed_previous,
				initial_state=initial_state,
				not_shared=self.not_shared)

		with vs.variable_scope('SRN_Model'):
			self.outputs, self.losses, self.state = model_utils.make_model(
					self.inputs, self.targets, self.weights,
					lambda x,y: Net(x,
								feed_previous=feed_previous,
								initial_state=y),
					softmax_loss_function=softmax_loss_function,
					initial_state=self.initial_state)
			
			if forward_only:
				if output_projection is not None:
					for b,output in enumerate(self.outputs):
						self.outputs[b] = tf.matmul(output, output_projection[0]) + output_projection[1]

		# Gradients and SGD update operation for training the model.
		params = tf.compat.v1.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			#opt = tf.train.AdamOptimizer(self.learning_rate)
			opt = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
			#opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, 
													self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.updates.append(opt.apply_gradients( 
				zip(clipped_gradients, params), global_step=self.global_step))
		self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=self.max_checkpoints_to_keep)

	def step(self, session, inputs, targets, weights, forward_only, initial_state=None):
		if len(self.inputs) != len(inputs):
			raise ValueError("self.inputs length and inputs length mismatch,"
						" %d != %d." % (len(self.inputs), len(inputs)))

		input_feed = {}
		for i in range(len(inputs)):
			input_feed[self.inputs[i].name] = inputs[i]
		for i in range(len(targets)):
			input_feed[self.targets[i].name] = targets[i]
		for i in range(len(weights)):
			input_feed[self.weights[i].name] = weights[i]
		if initial_state is not None:
			input_feed[self.initial_state.name] = initial_state
		else:
			input_feed[self.initial_state.name] = self.zero_state.eval()

		if not forward_only:
			output_feed = [self.updates[0],
							self.losses,
							self.state]
		else:
			output_feed = [self.outputs,
						   self.losses,
						   self.state]

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return None, outputs[1], outputs[2]
		else:
			return outputs[0], outputs[1], outputs[2]

"""Dependent output BRNN model
"""
class DBRNNModel(RNNModel):
	def __init__(self, config, mode, forward_only, cell_mode=None, no_previous=False, max_cell_length=None):
		super(DBRNNModel, self).__init__(config, mode, cell_mode=cell_mode, no_previous=no_previous,
										max_cell_length=max_cell_length)
		self.cell_fw = self.cell
		self.cell_bw = self.cell
		output_projection_forward = None
		output_projection_backward = None
		softmax_loss_function_forward = None
		softmax_loss_function_backward = None

		# forward output brnn sampled output projection
		with vs.variable_scope('forward_output_linear'):
			# sampled softmax
			if self.num_samples and self.num_samples < self.num_output_symbols:
				w_forward = tf.get_variable("Forward_proj_w", [self.cell_units, self.num_output_symbols])
				w_t_forward = tf.transpose(w_forward)
				b_forward = tf.get_variable("Forward_proj_b", [self.num_output_symbols])
				output_projection_forward = (w_forward, b_forward)
				def sampled_loss_forward(labels, inputs):
					labels = tf.reshape(labels, [-1, 1])
					local_w_t = tf.cast(w_t_forward, tf.float32)
					local_b = tf.cast(b_forward, tf.float32)
					local_inputs = tf.cast(inputs, tf.float32)
					return tf.nn.sampled_softmax_loss(
							weights=local_w_t,
							biases=local_b,
							labels=labels,
							inputs=local_inputs,
							num_sampled=self.num_samples,
							num_classes=self.num_output_symbols)
				softmax_loss_function_forward = sampled_loss_forward

		# backward output brnn sampled output projection
		with vs.variable_scope('backward_output_linear'):
			# sampled softmax
			if self.num_samples and self.num_samples < self.num_output_symbols:
				w_backward = tf.get_variable("Backward_proj_w", [self.cell_units, self.num_output_symbols])
				w_t_backward = tf.transpose(w_backward)
				b_backward = tf.get_variable("Backward_proj_b", [self.num_output_symbols])
				output_projection_backward = (w_backward, b_backward)
				def sampled_loss_backward(labels, inputs):
					labels = tf.reshape(labels, [-1, 1])
					local_w_t = tf.cast(w_t_backward, tf.float32)
					local_b = tf.cast(b_backward, tf.float32)
					local_inputs = tf.cast(inputs, tf.float32)
					return tf.nn.sampled_softmax_loss(
							weights=local_w_t,
							biases=local_b,
							labels=labels,
							inputs=local_inputs,
							num_sampled=self.num_samples,
							num_classes=self.num_output_symbols)
				softmax_loss_function_backward = sampled_loss_backward

		with vs.variable_scope('Dependent_BRNN_Model'):
			# make sampled softmax
			output_projection = None
			softmax_loss_function = None
			if self.num_samples and self.num_samples < self.num_output_symbols:
				w = tf.get_variable("proj_w", [self.cell_fw.output_size+self.cell_bw.output_size, self.num_output_symbols])
				w_t = tf.transpose(w)
				b = tf.get_variable("proj_b", [self.num_output_symbols])
				output_projection = (w, b)
				def sampled_loss(labels, inputs):
					labels = tf.reshape(labels, [-1, 1])
					local_w_t = tf.cast(w_t, tf.float32)
					local_b = tf.cast(b, tf.float32)
					local_inputs = tf.cast(inputs, tf.float32)
					return tf.nn.sampled_softmax_loss(
						weights=local_w_t,
						biases=local_b,
						labels=labels,
						inputs=local_inputs,
						num_sampled=self.num_samples,
						num_classes=self.num_output_symbols)
				softmax_loss_function = sampled_loss

			self.brnn_outputs, self.state = model_utils.dependent_brnn(
				self.inputs, self.cell_fw, self.cell_bw,
				num_input_symbols=self.num_input_symbols,
				num_output_symbols=self.num_output_symbols,
				embedding_size=self.embedding_size,
				output_projection_fw=output_projection_forward,
				output_projection_bw=output_projection_backward,
				not_shared=self.not_shared)

			self.losses_fw = model_utils.sequence_loss(self.brnn_outputs[0],
        										self.targets,
        										self.weights,
        										softmax_loss_function=softmax_loss_function_forward)
			self.losses_bw = model_utils.sequence_loss(self.brnn_outputs[1],
        										self.targets,
        										self.weights,
        										softmax_loss_function=softmax_loss_function_backward)
			# Combine the output
			self.outputs = []
			for time_step in xrange(len(self.brnn_outputs[0])):
				with vs.variable_scope(
          				vs.get_variable_scope(), reuse=True if time_step > 0 else None):
					self.outputs.append(model_utils.linear(array_ops.concat([self.brnn_outputs[0][time_step], self.brnn_outputs[1][time_step]], -1), 
														self.num_output_symbols,scope='output_projection'))
			self.losses = model_utils.sequence_loss(self.outputs,
        										self.targets,
        										self.weights,
        										softmax_loss_function=softmax_loss_function)

		# Gradients and SGD update operation for training the model.
		all_params = tf.compat.v1.trainable_variables()
		params_fw = [p for p in all_params if p.name.find('input_brnn')!=-1 or p.name.find('output_brnn/FW')!=-1]
		params_bw = [p for p in all_params if p.name.find('input_brnn')!=-1 or p.name.find('output_brnn/BW')!=-1]
		# shared, provides better performance
		params = all_params
		# not shared
		#params = [p for p in all_params if p not in params_fw and p not in params_bw]
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			#opt = tf.train.AdamOptimizer(self.learning_rate)
			opt = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			gradients_fw = tf.gradients(self.losses_fw, params_fw)
			gradients_bw = tf.gradients(self.losses_bw, params_bw)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, 
													self.max_gradient_norm)
			clipped_gradients_fw, norm_fw = tf.clip_by_global_norm(gradients_fw, 
													self.max_gradient_norm)
			clipped_gradients_bw, norm_bw = tf.clip_by_global_norm(gradients_bw, 
													self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.gradient_norms.append(norm_fw)
			self.gradient_norms.append(norm_bw)
			self.updates.append(opt.apply_gradients( 
				zip(clipped_gradients, params)))
			self.updates.append(opt.apply_gradients( 
				zip(clipped_gradients_fw, params_fw)))
			self.updates.append(opt.apply_gradients( 
				zip(clipped_gradients_bw, params_bw), global_step=self.global_step))
		self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=self.max_checkpoints_to_keep)

	def step(self, session, inputs, targets, weights, forward_only):
		if len(self.inputs) != len(inputs):
			raise ValueError("self.inputs length and inputs length mismatch,"
						" %d != %d." % (len(self.inputs), len(inputs)))

		input_feed = {}
		for i in range(len(inputs)):
			input_feed[self.inputs[i].name] = inputs[i]
		for i in range(len(targets)):
			input_feed[self.targets[i].name] = targets[i]
		for i in range(len(weights)):
			input_feed[self.weights[i].name] = weights[i]

		if not forward_only:
			output_feed = [self.updates[0],
							self.updates[1],
							self.updates[2],
							self.losses,
							self.losses_fw,
							self.losses_bw,
							self.state]
		else:
			if self.mode == 'train':
				output_feed = [self.outputs,
							   self.losses,
							   [self.losses_fw, self.losses_bw]]
			else:
				output_feed = [self.outputs,
							   self.brnn_outputs,
							   [self.state, self.losses, self.losses_fw, self.losses_bw]]

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[5], outputs[3], outputs[4]
		else:
			return outputs[0], outputs[1], outputs[2]

class ENC_DEC_ATTModel(RNNModel):
	def __init__(self, config, mode, forward_only, feed_previous, cell_mode=None, no_previous=False, max_cell_length=None):
		super(ENC_DEC_ATTModel, self).__init__(config, mode, cell_mode=cell_mode, no_previous=no_previous, max_cell_length=max_cell_length)

		self.decoder_inputs = []
		for i in xrange(self.max_output_seq_length):
			self.decoder_inputs.append(tf.compat.v1.placeholder(dtype=tf.int64,
	                                  shape=[None],  # batch_size
	                                  name="decoder_input_feed{0}".format(i)))

		# make sampled softmax
		output_projection = None
		softmax_loss_function = None
		if self.num_samples and self.num_samples < self.num_output_symbols:
			w = tf.get_variable("proj_w", [self.cell_units, self.num_output_symbols])
			w_t = tf.transpose(w)
			b = tf.get_variable("proj_b", [self.num_output_symbols])
			output_projection = (w, b)
			def sampled_loss(labels, inputs):
				labels = tf.reshape(labels, [-1, 1])
				local_w_t = tf.cast(w_t, tf.float32)
				local_b = tf.cast(b, tf.float32)
				local_inputs = tf.cast(inputs, tf.float32)
				return tf.nn.sampled_softmax_loss(
						weights=local_w_t,
						biases=local_b,
						labels=labels,
						inputs=local_inputs,
						num_sampled=self.num_samples,
						num_classes=self.num_output_symbols)
			softmax_loss_function = sampled_loss

		# one to one learning task
		def Net(encoder_inputs, decoder_inputs, feed_previous):
			return model_utils.embedding_attention_seq2seq(
				encoder_inputs, decoder_inputs, self.cell,
				num_encoder_symbols=self.num_input_symbols,
				num_decoder_symbols=self.num_output_symbols,
				embedding_size=self.embedding_size,
				output_projection=output_projection,
				feed_previous=feed_previous,
				not_shared=self.not_shared)

		self.outputs, self.losses, self.state = model_utils.make_model(
				self.inputs, self.targets, self.weights,
				lambda x,y: Net(x, y,
							feed_previous=feed_previous),
				softmax_loss_function=softmax_loss_function,
				initial_state=self.initial_state,
				decoder_inputs=self.decoder_inputs)
		if forward_only:
			if output_projection is not None:
				for b,output in enumerate(self.outputs):
					self.outputs[b] = tf.matmul(output, output_projection[0]) + output_projection[1]

		# Gradients and SGD update operation for training the model.
		params = tf.compat.v1.trainable_variables()
		if not forward_only:
			self.gradient_norms = []
			self.updates = []
			#opt = tf.train.AdamOptimizer(self.learning_rate)
			opt = tf.compat.v1.train.AdagradOptimizer(self.learning_rate)
			#opt = tf.train.GradientDescentOptimizer(self.learning_rate)
			gradients = tf.gradients(self.losses, params)
			clipped_gradients, norm = tf.clip_by_global_norm(gradients, 
													self.max_gradient_norm)
			self.gradient_norms.append(norm)
			self.updates.append(opt.apply_gradients( 
				zip(clipped_gradients, params), global_step=self.global_step))
		self.saver = tf.compat.v1.train.Saver(tf.compat.v1.global_variables(), max_to_keep=self.max_checkpoints_to_keep)

	def step(self, session, inputs, targets, weights, forward_only, initial_state=None):
		if len(self.inputs) != len(inputs):
			raise ValueError("self.inputs length and inputs length mismatch,"
						" %d != %d." % (len(self.inputs), len(inputs)))

		input_feed = {}
		for i in range(len(inputs)):
			input_feed[self.inputs[i].name] = inputs[i]
		input_feed[self.decoder_inputs[0].name] = [data_utils.BOS_ID for i in range(len(targets[0]))]
		for i in range(len(targets)):
			input_feed[self.targets[i].name] = targets[i]
			if i < len(targets)-1:
				input_feed[self.decoder_inputs[i+1].name] = targets[i]
		for i in range(len(weights)):
			input_feed[self.weights[i].name] = weights[i]
		if initial_state is not None:
			input_feed[self.initial_state.name] = initial_state
		else:
			input_feed[self.initial_state.name] = self.zero_state.eval()

		if not forward_only:
			output_feed = [self.updates[0],
							self.outputs,
							self.losses,
							self.state]
		else:
			output_feed = [self.outputs,
						   self.losses,
						   self.state]

		outputs = session.run(output_feed, input_feed)
		if not forward_only:
			return outputs[1], outputs[2], outputs[3]
		else:
			return outputs[0], outputs[1], outputs[2]