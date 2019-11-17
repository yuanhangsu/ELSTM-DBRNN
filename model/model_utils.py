from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# We disable pylint because we need python3 compatibility.
from six.moves import xrange  # pylint: disable=redefined-builtin
from six.moves import zip  # pylint: disable=redefined-builtin
import math

import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest

linear = tf.contrib.layers.fully_connected

class OutputProjectionWrapper(tf.compat.v1.nn.rnn_cell.RNNCell):
	"""Operator adding an output projection to the given cell.

	Note: in many cases it may be more efficient to not use this wrapper,
	but instead concatenate the whole sequence of your outputs in time,
	do the projection on this batch-concatenated sequence, then split it
	if needed or directly feed into a softmax.
	"""

	def __init__(self, cell, output_size):
		"""Create a cell with output projection.

		Args:
			cell: an RNNCell, a projection to output_size is added to it.
			output_size: integer, the size of the output after projection.

		Raises:
			TypeError: if cell is not an RNNCell.
			ValueError: if output_size is not positive.
		"""
		if not isinstance(cell, tf.compat.v1.nn.rnn_cell.RNNCell):
			raise TypeError("The parameter cell is not RNNCell.")
		if output_size < 1:
			raise ValueError("Parameter output_size must be > 0: %d." % output_size)
		self._cell = cell
		self._output_size = output_size

	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._output_size

	def num_units(self):
		return self._cell.num_units

	def __call__(self, inputs, state, ab=None, scope=None):
		"""Run the cell and output projection on inputs, starting from state."""
		if ab:
			output, res_state = self._cell(inputs, state, ab=ab)      
		else:
			output, res_state = self._cell(inputs, state)
		# Default scope: "OutputProjectionWrapper"
		with variable_scope.variable_scope(scope or type(self).__name__):
			projected = linear(output, self._output_size)
		return projected, res_state

class EmbeddingWrapper(tf.compat.v1.nn.rnn_cell.RNNCell):
	"""Operator adding input embedding to the given cell.

	Note: in many cases it may be more efficient to not use this wrapper,
	but instead concatenate the whole sequence of your inputs in time,
	do the embedding on this batch-concatenated sequence, then split it and
	feed into your RNN.
	"""

	def __init__(self, cell, embedding_classes, embedding_size, initializer=None):
		"""Create a cell with an added input embedding.

		Args:
			cell: an RNNCell, an embedding will be put before its inputs.
			embedding_classes: integer, how many symbols will be embedded.
			embedding_size: integer, the size of the vectors we embed into.
			initializer: an initializer to use when creating the embedding;
				if None, the initializer from variable scope or a default one is used.

		Raises:
			TypeError: if cell is not an RNNCell.
			ValueError: if embedding_classes is not positive.
		"""
		if not isinstance(cell, tf.compat.v1.nn.rnn_cell.RNNCell):
			raise TypeError("The parameter cell is not RNNCell.")
		if embedding_classes <= 0 or embedding_size <= 0:
			raise ValueError("Both embedding_classes and embedding_size must be > 0: "
											 "%d, %d." % (embedding_classes, embedding_size))
		self._cell = cell
		self._embedding_classes = embedding_classes
		self._embedding_size = embedding_size
		self._initializer = initializer

	@property
	def state_size(self):
		return self._cell.state_size

	@property
	def output_size(self):
		return self._cell.output_size

	@property
	def num_units(self):
		return self._cell.num_units

	def __call__(self, inputs, state, ab=None, scope=None):
		"""Run the cell on embedded inputs."""
		with variable_scope.variable_scope(scope or type(self).__name__):  # "EmbeddingWrapper"
			with ops.device("/cpu:0"):
				if self._initializer:
					initializer = self._initializer
				elif variable_scope.get_variable_scope().initializer:
					initializer = variable_scope.get_variable_scope().initializer
				else:
					# Default initializer for embeddings should have variance=1.
					sqrt3 = math.sqrt(3)  # Uniform(-sqrt(3), sqrt(3)) has variance=1.
					initializer = init_ops.random_uniform_initializer(-sqrt3, sqrt3)

				if type(state) is tuple:
					data_type = state[0].dtype
				else:
					data_type = state.dtype

				embedding = variable_scope.get_variable(
						"embedding", [self._embedding_classes, self._embedding_size],
						initializer=initializer,
						dtype=data_type)
				embedded = embedding_ops.embedding_lookup(
						embedding, array_ops.reshape(inputs, [-1]))
		if ab:
			return self._cell(embedded, state, ab=ab)
		else:
			return self._cell(embedded, state)

def basic_rnn(
	inputs,
	cell,
	num_input_symbols,
	num_output_symbols,
	embedding_size,
	output_projection=None,
	feed_previous=False,
	initial_state=None,
	dtype=dtypes.float32,
	not_shared=False,
	scope=None):
	with variable_scope.variable_scope(scope or "basic_rnn") as scope:
		if output_projection is not None:
			dtype = scope.dtype
			proj_weights = ops.convert_to_tensor(output_projection[0], dtype=dtype)
			proj_weights.get_shape().assert_is_compatible_with([None, num_output_symbols])
			proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
			proj_biases.get_shape().assert_is_compatible_with([num_output_symbols])
		else:
			cell = OutputProjectionWrapper(cell, num_output_symbols)
		embedding = None
		loop_function = None
		emb_inp = inputs
		if embedding_size > 0:
			embedding = variable_scope.get_variable("embedding", [num_input_symbols, embedding_size])
			if feed_previous:
				loop_function = _extract_argmax_and_embed(
						embedding, output_projection,# feed previous only when inferencing
						update_embedding=True) 
			emb_inp = (embedding_ops.embedding_lookup(embedding, i)
								 for i in inputs)
		
		return rnn_decoder(
				emb_inp, initial_state, cell, loop_function=loop_function, not_shared=not_shared)

"""brnn followed by basic rnn for dependent outputs
"""
def dependent_brnn(inputs, cell_fw, cell_bw,
	num_input_symbols,
	num_output_symbols,
	embedding_size,
	initial_state_fw=None, initial_state_bw=None,
	output_projection_fw=None,
	output_projection_bw=None,
	dtype=dtypes.float32, not_shared=False,
	scope=None):
	with variable_scope.variable_scope(scope or "dependent_brnn") as scope:
		em_cell_fw, em_cell_bw = cell_fw, cell_bw
		if embedding_size:
			# Forward input embedding
			em_cell_fw = EmbeddingWrapper(
											cell_fw,
											embedding_classes=num_input_symbols,
											embedding_size=embedding_size)
			# Backward input embedding
			em_cell_bw = EmbeddingWrapper(
											cell_bw,
											embedding_classes=num_input_symbols,
											embedding_size=embedding_size)

		# [time][batch][cell_fw.output_size + cell_bw.output_size]
		tmp, _, _ = bidirectional_rnn(em_cell_fw, em_cell_bw, inputs,
																	initial_state_fw=initial_state_fw, 
																	initial_state_bw=initial_state_bw,
																	dtype=dtype, not_shared=not_shared,
																	scope='input_brnn')

		""" basic rnn for dependent outputs
		each element in tmp is independent, to correlate them,
		it needs to be fed into another brnn
		"""
		brnn_outputs, state_fw, state_bw = bidirectional_rnn(
																		cell_fw, cell_bw, tmp,
																		initial_state_fw=initial_state_fw, 
																		initial_state_bw=initial_state_bw,
																		flat_outputs=False,
																		dtype=dtypes.float32, not_shared=not_shared,
																		scope='output_brnn')

		# output projection
		brnn_outputs_fw, brnn_outputs_bw = brnn_outputs
		outputs_fw, outputs_bw = [],[]
		if output_projection_fw is None:
			for time_step in xrange(len(brnn_outputs_fw)):
				with variable_scope.variable_scope(
					variable_scope.get_variable_scope(), reuse=True if time_step > 0 else None):
					outputs_fw.append(linear(brnn_outputs_fw[time_step],
																	num_output_symbols, scope="output_brnn/FW/linear"))
		if output_projection_bw is None:
			for time_step in xrange(len(brnn_outputs_bw)):
				with variable_scope.variable_scope(
					variable_scope.get_variable_scope(), reuse=True if time_step > 0 else None):
					outputs_bw.append(linear(brnn_outputs_bw[time_step],
																	num_output_symbols, scope="output_brnn/BW/linear"))
				
		outputs = (outputs_fw or brnn_outputs_fw, 
								outputs_bw or brnn_outputs_bw)
		
	return outputs, (state_fw, state_bw)

def sequence_loss_by_example(
	logits,
	targets,
	weights,
	average_across_timesteps=True,
	softmax_loss_function=None,
	name=None):
	"""Weighted cross-entropy loss for a sequence of logits (per example).

	Args:
		logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
		targets: List of 1D batch-sized int32 Tensors of the same length as logits.
		weights: List of 1D batch-sized float-Tensors of the same length as logits.
		average_across_timesteps: If set, divide the returned cost by the total
			label weight.
		softmax_loss_function: Function (labels-batch, inputs-batch) -> loss-batch
			to be used instead of the standard softmax (the default if this is None).
		name: Optional name for this operation, default: "sequence_loss_by_example".

	Returns:
		1D batch-sized float Tensor: The log-perplexity for each sequence.

	Raises:
		ValueError: If len(logits) is different from len(targets) or len(weights).
	"""
	if len(targets) != len(logits) or len(weights) != len(logits):
		raise ValueError("Lengths of logits, weights, and targets must be the same "
										 "%d, %d, %d." % (len(logits), len(weights), len(targets)))
	with ops.name_scope(name, "sequence_loss_by_example",
											logits + targets + weights):
		log_perp_list = []
		for logit, target, weight in zip(logits, targets, weights):
			if softmax_loss_function is None:
				# TODO(irving,ebrevdo): This reshape is needed because
				# sequence_loss_by_example is called with scalars sometimes, which
				# violates our general scalar strictness policy.
				target = array_ops.reshape(target, [-1])
				crossent = nn_ops.sparse_softmax_cross_entropy_with_logits(
						labels=target, logits=logit)
			else:
				crossent = softmax_loss_function(target, logit)
			log_perp_list.append(crossent * weight)
		log_perps = math_ops.add_n(log_perp_list)
		if average_across_timesteps:
			total_size = math_ops.add_n(weights)
			total_size += 1e-12  # Just to avoid division by 0 for all-0 weights.
			log_perps /= total_size
	return log_perps


def sequence_loss(
	logits,
	targets,
	weights,
	average_across_timesteps=True,
	average_across_batch=True,
	softmax_loss_function=None,
	name=None):
	"""Weighted cross-entropy loss for a sequence of logits, batch-collapsed.

	Args:
		logits: List of 2D Tensors of shape [batch_size x num_decoder_symbols].
		targets: List of 1D batch-sized int32 Tensors of the same length as logits.
		weights: List of 1D batch-sized float-Tensors of the same length as logits.
		average_across_timesteps: If set, divide the returned cost by the total
			label weight.
		average_across_batch: If set, divide the returned cost by the batch size.
		softmax_loss_function: Function (inputs-batch, labels-batch) -> loss-batch
			to be used instead of the standard softmax (the default if this is None).
		name: Optional name for this operation, defaults to "sequence_loss".

	Returns:
		A scalar float Tensor: The average log-perplexity per symbol (weighted).

	Raises:
		ValueError: If len(logits) is different from len(targets) or len(weights).
	"""
	with ops.name_scope(name, "sequence_loss", logits + targets + weights):
		cost = math_ops.reduce_sum(
				sequence_loss_by_example(
						logits,
						targets,
						weights,
						average_across_timesteps=average_across_timesteps,
						softmax_loss_function=softmax_loss_function))
		if average_across_batch:
			batch_size = array_ops.shape(targets[0])[0]
			return cost / math_ops.cast(batch_size, cost.dtype)
		else:
			return cost

def make_model(
	encoder_inputs,
	targets,
	weights,
	Net,
	softmax_loss_function=None,
	initial_state=None,
	decoder_inputs=None,
	per_example_loss=False,
	name=None):
	if decoder_inputs:
		all_inputs = encoder_inputs + decoder_inputs + targets + weights
	else:
		all_inputs = encoder_inputs + targets + weights
	with ops.name_scope(name, "make_model", all_inputs):
		with variable_scope.variable_scope(
				variable_scope.get_variable_scope(), reuse=None):
			if decoder_inputs:
				outputs, state = Net(encoder_inputs,
												decoder_inputs)
			elif initial_state is not None:
				outputs, state = Net(encoder_inputs, 
												initial_state)
			else:
				outputs, state = Net(encoder_inputs)
			if per_example_loss:
				losses = sequence_loss_by_example(
										outputs,
										targets,
										weights,
										softmax_loss_function=softmax_loss_function)
			else:
				losses = sequence_loss(
										outputs,
										targets,
										weights,
										softmax_loss_function=softmax_loss_function)
	return outputs, losses, state

def _extract_argmax_and_embed(
	embedding,
	output_projection=None,
	update_embedding=True):
	"""Get a loop_function that extracts the previous symbol and embeds it.

	Args:
		embedding: embedding tensor for symbols.
		output_projection: None or a pair (W, B). If provided, each fed previous
			output will first be multiplied by W and added B.
		update_embedding: Boolean; if False, the gradients will not propagate
			through the embeddings.

	Returns:
		A loop function.
	"""

	def loop_function(prev, _):
		if output_projection is not None:
			prev = nn_ops.xw_plus_b(prev, output_projection[0], output_projection[1])
		prev_symbol = math_ops.argmax(prev, 1)
		# Note that gradients will not propagate through the second parameter of
		# embedding_lookup.
		emb_prev = embedding_ops.embedding_lookup(embedding, prev_symbol)
		if not update_embedding:
			emb_prev = array_ops.stop_gradient(emb_prev)
		return emb_prev

	return loop_function

def _rnn(cell, inputs, initial_state=None, dtype=None,
				sequence_length=None, not_shared = False, scope=None):

	"""This rnn allows some cell parameters not shared across time
	set not_shared to be True
	"""
	if not nest.is_sequence(inputs):
		raise TypeError("inputs must be a sequence")
	if not inputs:
		raise ValueError("inputs must not be empty")

	outputs = []
	# Create a new scope in which the caching device is either
	# determined by the parent scope, or is set to place the cached
	# Variable using the same placement as for the rest of the RNN.
	with variable_scope.variable_scope(scope or "RNN") as varscope:
		if varscope.caching_device is None:
			varscope.set_caching_device(lambda op: op.device)

		# Obtain the first sequence of the input
		first_input = inputs
		while nest.is_sequence(first_input):
			first_input = first_input[0]

		# Temporarily avoid EmbeddingWrapper and seq2seq badness
		# TODO(lukaszkaiser): remove EmbeddingWrapper
		if first_input.get_shape().ndims != 1:

			input_shape = first_input.get_shape().with_rank_at_least(2)
			fixed_batch_size = input_shape[0]

			flat_inputs = nest.flatten(inputs)
			for flat_input in flat_inputs:
				input_shape = flat_input.get_shape().with_rank_at_least(2)
				batch_size, input_size = input_shape[0], input_shape[1:]
				fixed_batch_size.merge_with(batch_size)
				for i, size in enumerate(input_size):
					if size.value is None:
						raise ValueError(
								"Input size (dimension %d of inputs) must be accessible via "
								"shape inference, but saw value None." % i)
		else:
			fixed_batch_size = first_input.get_shape().with_rank_at_least(1)[0]

		if fixed_batch_size.value:
			batch_size = fixed_batch_size.value
		else:
			batch_size = array_ops.shape(first_input)[0]
		if initial_state is not None:
			state = initial_state
		else:
			if not dtype:
				raise ValueError("If no initial_state is provided, "
												 "dtype must be specified")
			state = cell.zero_state(batch_size, dtype)

		if sequence_length is not None:  # Prepare variables
			sequence_length = ops.convert_to_tensor(
					sequence_length, name="sequence_length")
			if sequence_length.get_shape().ndims not in (None, 1):
				raise ValueError(
						"sequence_length must be a vector of length batch_size")
			def _create_zero_output(output_size):
				# convert int to TensorShape if necessary
				size = _state_size_with_prefix(output_size, prefix=[batch_size])
				output = array_ops.zeros(
						array_ops.pack(size), _infer_state_dtype(dtype, state))
				shape = _state_size_with_prefix(
						output_size, prefix=[fixed_batch_size.value])
				output.set_shape(tensor_shape.TensorShape(shape))
				return output

			output_size = cell.output_size
			flat_output_size = nest.flatten(output_size)
			flat_zero_output = tuple(
					_create_zero_output(size) for size in flat_output_size)
			zero_output = nest.pack_sequence_as(structure=output_size,
																					flat_sequence=flat_zero_output)

			sequence_length = math_ops.to_int32(sequence_length)
			min_sequence_length = math_ops.reduce_min(sequence_length)
			max_sequence_length = math_ops.reduce_max(sequence_length)

		for time, input_ in enumerate(inputs):
			ab = []
			with variable_scope.variable_scope("ELSTM_s", reuse=None if time<not_shared else True):
				if not_shared:
					ab.append(tf.get_variable(
								"ab_a{0}".format(time%not_shared), initializer=tf.ones_initializer(dtype=tf.float32), shape=[cell.num_units], dtype=tf.float32))       
			with tf.variable_scope('RNN_loop',reuse=True if time>0 else None):
				# pylint: disable=cell-var-from-loop
				if ab:
					call_cell = lambda: cell(input_, state, ab=ab)
				else:
					call_cell = lambda: cell(input_, state)
				# pylint: enable=cell-var-from-loop
				(output, state) = call_cell()

				outputs.append(output)

		return (outputs, state)	

def bidirectional_rnn(cell_fw, cell_bw, inputs,
											initial_state_fw=None, initial_state_bw=None,
											flat_outputs=True,
											dtype=None, sequence_length=None, 
											not_shared=False, scope=None):
	"""Creates a bidirectional recurrent neural network.

	Similar to the unidirectional case above (rnn) but takes input and builds
	independent forward and backward RNNs with the final forward and backward
	outputs depth-concatenated, such that the output will have the format
	[time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
	forward and backward cell must match. The initial state for both directions
	is zero by default (but can be set optionally) and no intermediate states are
	ever returned -- the network is fully unrolled for the given (passed in)
	length(s) of the sequence(s) or completely unrolled if length(s) is not given.

	Args:
		cell_fw: An instance of RNNCell, to be used for forward direction.
		cell_bw: An instance of RNNCell, to be used for backward direction.
		inputs: A length T list of inputs, each a tensor of shape
			[batch_size, input_size], or a nested tuple of such elements.
		initial_state_fw: (optional) An initial state for the forward RNN.
			This must be a tensor of appropriate type and shape
			`[batch_size, cell_fw.state_size]`.
			If `cell_fw.state_size` is a tuple, this should be a tuple of
			tensors having shapes `[batch_size, s] for s in cell_fw.state_size`.
		initial_state_bw: (optional) Same as for `initial_state_fw`, but using
			the corresponding properties of `cell_bw`.
		dtype: (optional) The data type for the initial state.  Required if
			either of the initial states are not provided.
		sequence_length: (optional) An int32/int64 vector, size `[batch_size]`,
			containing the actual lengths for each of the sequences.
		scope: VariableScope for the created subgraph; defaults to "BiRNN"

	Returns:
		A tuple (outputs, output_state_fw, output_state_bw) where:
			outputs is a length `T` list of outputs (one for each input), which
				are depth-concatenated forward and backward outputs.
			output_state_fw is the final state of the forward rnn.
			output_state_bw is the final state of the backward rnn.

	Raises:
		TypeError: If `cell_fw` or `cell_bw` is not an instance of `RNNCell`.
		ValueError: If inputs is None or an empty list.
	"""

	if not isinstance(cell_fw, tf.compat.v1.nn.rnn_cell.RNNCell):
		raise TypeError("cell_fw must be an instance of RNNCell")
	if not isinstance(cell_bw, tf.compat.v1.nn.rnn_cell.RNNCell):
		raise TypeError("cell_bw must be an instance of RNNCell")
	if not nest.is_sequence(inputs):
		raise TypeError("inputs must be a sequence")
	if not inputs:
		raise ValueError("inputs must not be empty")

	with variable_scope.variable_scope(scope or "BiRNN"):
		# Forward direction
		with variable_scope.variable_scope("FW") as fw_scope:
			output_fw, output_state_fw = _rnn(cell_fw, inputs, initial_state_fw, dtype,
																			 sequence_length, not_shared=not_shared, scope=fw_scope)

		# Backward direction
		with variable_scope.variable_scope("BW") as bw_scope:
			reversed_inputs = _reverse_seq(inputs, sequence_length)
			tmp, output_state_bw = _rnn(cell_bw, reversed_inputs, initial_state_bw,
																 dtype, sequence_length, not_shared=not_shared, scope=bw_scope)

	output_bw = _reverse_seq(tmp, sequence_length)
	if flat_outputs:
		# Concat each of the forward/backward outputs
		flat_output_fw = nest.flatten(output_fw)
		flat_output_bw = nest.flatten(output_bw)

		flat_outputs = tuple(array_ops.concat([fw, bw],1)
												 for fw, bw in zip(flat_output_fw, flat_output_bw))

		outputs = nest.pack_sequence_as(structure=output_fw,
																		flat_sequence=flat_outputs)
	else:
		outputs = (output_fw, output_bw)

	return (outputs, output_state_fw, output_state_bw)

def _reverse_seq(input_seq, lengths):
	"""Reverse a list of Tensors up to specified lengths.

	Args:
		input_seq: Sequence of seq_len tensors of dimension (batch_size, n_features)
							 or nested tuples of tensors.
		lengths:   A `Tensor` of dimension batch_size, containing lengths for each
							 sequence in the batch. If "None" is specified, simply reverses
							 the list.

	Returns:
		time-reversed sequence
	"""
	if lengths is None:
		return list(reversed(input_seq))

	flat_input_seq = tuple(nest.flatten(input_) for input_ in input_seq)

	flat_results = [[] for _ in range(len(input_seq))]
	for sequence in zip(*flat_input_seq):
		input_shape = tensor_shape.unknown_shape(
				ndims=sequence[0].get_shape().ndims)
		for input_ in sequence:
			input_shape.merge_with(input_.get_shape())
			input_.set_shape(input_shape)

		# Join into (time, batch_size, depth)
		s_joined = array_ops.pack(sequence)

		# TODO(schuster, ebrevdo): Remove cast when reverse_sequence takes int32
		if lengths is not None:
			lengths = math_ops.to_int64(lengths)

		# Reverse along dimension 0
		s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
		# Split again into list
		result = array_ops.unpack(s_reversed)
		for r, flat_result in zip(result, flat_results):
			r.set_shape(input_shape)
			flat_result.append(r)

	results = [nest.pack_sequence_as(structure=input_, flat_sequence=flat_result)
						 for input_, flat_result in zip(input_seq, flat_results)]
	return results

def rnn_decoder(decoder_inputs,
								initial_state,
								cell,
								loop_function=None,
								not_shared=False,
								scope=None):
	"""RNN decoder for the sequence-to-sequence model.

	Args:
		decoder_inputs: A list of 2D Tensors [batch_size x input_size].
		initial_state: 2D Tensor with shape [batch_size x cell.state_size].
		cell: tf.compat.v1.nn.rnn_cell.RNNCell defining the cell function and size.
		loop_function: If not None, this function will be applied to the i-th output
			in order to generate the i+1-st input, and decoder_inputs will be ignored,
			except for the first element ("GO" symbol). This can be used for decoding,
			but also for training to emulate http://arxiv.org/abs/1506.03099.
			Signature -- loop_function(prev, i) = next
				* prev is a 2D Tensor of shape [batch_size x output_size],
				* i is an integer, the step number (when advanced control is needed),
				* next is a 2D Tensor of shape [batch_size x input_size].
		scope: VariableScope for the created subgraph; defaults to "rnn_decoder".

	Returns:
		A tuple of the form (outputs, state), where:
			outputs: A list of the same length as decoder_inputs of 2D Tensors with
				shape [batch_size x output_size] containing generated outputs.
			state: The state of each cell at the final time-step.
				It is a 2D Tensor of shape [batch_size x cell.state_size].
				(Note that in some cases, like basic RNN cell or GRU cell, outputs and
				 states can be the same. They are different for LSTM cells though.)
	"""
	with variable_scope.variable_scope(scope or "rnn_decoder"):
		state = initial_state
		outputs = []
		prev = None
		for i, inp in enumerate(decoder_inputs):
			if loop_function is not None and prev is not None:
				with variable_scope.variable_scope("loop_function", reuse=True):
					inp = loop_function(prev, i)

			with variable_scope.variable_scope("ELSTM_s", reuse=None if i<not_shared else True):
				ab = []
				if not_shared:
					ab.append(tf.get_variable(
								"ab_a{0}".format(i%not_shared), initializer=tf.ones_initializer(dtype=tf.float32), shape=[cell.num_units()], dtype=tf.float32))

			with variable_scope.variable_scope("rnn_loop", reuse=True if i>0 else None):
				output, state = cell(inp, state, ab=ab)

				outputs.append(output)
				if loop_function is not None:
					prev = output

				#activations.append(tf.get_variable())

	return (outputs, state)	

def attention_decoder(
	decoder_inputs,
	initial_state,
	attention_states,
	cell,
	output_size=None,
	num_heads=1,
	loop_function=None,
	dtype=None,
	not_shared=False,
	scope=None,
	initial_state_attention=False):
	"""RNN decoder with attention for the sequence-to-sequence model.

	In this context "attention" means that, during decoding, the RNN can look up
	information in the additional tensor attention_states, and it does this by
	focusing on a few entries from the tensor. This model has proven to yield
	especially good results in a number of sequence-to-sequence tasks. This
	implementation is based on http://arxiv.org/abs/1412.7449 (see below for
	details). It is recommended for complex sequence-to-sequence tasks.

	Args:
		decoder_inputs: A list of 2D Tensors [batch_size x input_size].
		initial_state: 2D Tensor [batch_size x cell.state_size].
		attention_states: 3D Tensor [batch_size x attn_length x attn_size].
		cell: core_rnn_cell.RNNCell defining the cell function and size.
		output_size: Size of the output vectors; if None, we use cell.output_size.
		num_heads: Number of attention heads that read from attention_states.
		loop_function: If not None, this function will be applied to i-th output
			in order to generate i+1-th input, and decoder_inputs will be ignored,
			except for the first element ("GO" symbol). This can be used for decoding,
			but also for training to emulate http://arxiv.org/abs/1506.03099.
			Signature -- loop_function(prev, i) = next
				* prev is a 2D Tensor of shape [batch_size x output_size],
				* i is an integer, the step number (when advanced control is needed),
				* next is a 2D Tensor of shape [batch_size x input_size].
		dtype: The dtype to use for the RNN initial state (default: tf.float32).
		scope: VariableScope for the created subgraph; default: "attention_decoder".
		initial_state_attention: If False (default), initial attentions are zero.
			If True, initialize the attentions from the initial state and attention
			states -- useful when we wish to resume decoding from a previously
			stored decoder state and attention states.

	Returns:
		A tuple of the form (outputs, state), where:
			outputs: A list of the same length as decoder_inputs of 2D Tensors of
				shape [batch_size x output_size]. These represent the generated outputs.
				Output i is computed from input i (which is either the i-th element
				of decoder_inputs or loop_function(output {i-1}, i)) as follows.
				First, we run the cell on a combination of the input and previous
				attention masks:
					cell_output, new_state = cell(linear(input, prev_attn), prev_state).
				Then, we calculate new attention masks:
					new_attn = softmax(V^T * tanh(W * attention_states + U * new_state))
				and then we calculate the output:
					output = linear(cell_output, new_attn).
			state: The state of each decoder cell the final time-step.
				It is a 2D Tensor of shape [batch_size x cell.state_size].

	Raises:
		ValueError: when num_heads is not positive, there are no inputs, shapes
			of attention_states are not set, or input size cannot be inferred
			from the input.
	"""
	if not decoder_inputs:
		raise ValueError("Must provide at least 1 input to attention decoder.")
	if num_heads < 1:
		raise ValueError("With less than 1 heads, use a non-attention decoder.")
	if attention_states.get_shape()[2].value is None:
		raise ValueError("Shape[2] of attention_states must be known: %s" %
										 attention_states.get_shape())
	if output_size is None:
		output_size = cell.output_size

	with variable_scope.variable_scope(
			scope or "attention_decoder", dtype=dtype) as scope:
		dtype = scope.dtype

		batch_size = array_ops.shape(decoder_inputs[0])[0]  # Needed for reshaping.
		attn_length = attention_states.get_shape()[1].value
		if attn_length is None:
			attn_length = array_ops.shape(attention_states)[1]
		attn_size = attention_states.get_shape()[2].value

		# To calculate W1 * h_t we use a 1-by-1 convolution, need to reshape before.
		hidden = array_ops.reshape(attention_states,
															 [-1, attn_length, 1, attn_size])
		hidden_features = []
		v = []
		attention_vec_size = attn_size  # Size of query vectors for attention.
		for a in xrange(num_heads):
			k = variable_scope.get_variable("AttnW_%d" % a,
																			[1, 1, attn_size, attention_vec_size])
			hidden_features.append(nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME"))
			v.append(
					variable_scope.get_variable("AttnV_%d" % a, [attention_vec_size]))

		state = initial_state

		def attention(query):
			"""Put attention masks on hidden using hidden_features and query."""
			ds = []  # Results of attention reads will be stored here.
			if nest.is_sequence(query):  # If the query is a tuple, flatten it.
				query_list = nest.flatten(query)
				for q in query_list:  # Check that ndims == 2 if specified.
					ndims = q.get_shape().ndims
					if ndims:
						assert ndims == 2
				query = array_ops.concat(query_list, 1)
			for a in xrange(num_heads):
				with variable_scope.variable_scope("Attention_%d" % a):
					y = linear(query, attention_vec_size)
					y = array_ops.reshape(y, [-1, 1, 1, attention_vec_size])
					# Attention mask is a softmax of v^T * tanh(...).
					s = math_ops.reduce_sum(v[a] * math_ops.tanh(hidden_features[a] + y),
																	[2, 3])
					a = nn_ops.softmax(s)
					# Now calculate the attention-weighted vector d.
					d = math_ops.reduce_sum(
							array_ops.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2])
					ds.append(array_ops.reshape(d, [-1, attn_size]))
			return ds

		outputs = []
		prev = None
		batch_attn_size = array_ops.stack([batch_size, attn_size])
		attns = [
				array_ops.zeros(
						batch_attn_size, dtype=dtype) for _ in xrange(num_heads)
		]
		for a in attns:  # Ensure the second shape of attention vectors is set.
			a.set_shape([None, attn_size])
		if initial_state_attention:
			attns = attention(initial_state)
		
		for i, inp in enumerate(decoder_inputs):
			with variable_scope.variable_scope("ELSTM_s", reuse=None if i<not_shared else True):
				ab = []
				if not_shared:
					ab.append(tf.get_variable(
								"ab_a{0}".format(i%not_shared), initializer=tf.ones_initializer(dtype=tf.float32), shape=[cell.num_units()], dtype=tf.float32))
			#if i > 0:
			#  variable_scope.get_variable_scope().reuse_variables()

			with variable_scope.variable_scope("attention_decoder_loop", reuse=True if i>0 else None):
				# If loop_function is set, we use it instead of decoder_inputs.
				if loop_function is not None and prev is not None:
					with variable_scope.variable_scope("loop_function", reuse=True):
							inp = loop_function(prev, i)
				# Merge input and previous attentions into one vector of the right size.
				input_size = inp.get_shape().with_rank(2)[1]
				if input_size.value is None:
					raise ValueError("Could not infer input size from input: %s" % inp.name)
				x = linear(array_ops.concat([inp]+attns, -1), int(input_size))
				# Run the RNN.    
				cell_output, state = cell(x, state, ab=ab)
				# Run the attention mechanism.
				if i == 0 and initial_state_attention:
					with variable_scope.variable_scope(
							variable_scope.get_variable_scope(), reuse=True):
							attns = attention(state)
				else:
					attns = attention(state)

				with variable_scope.variable_scope("AttnOutputProjection"):
					output = linear(array_ops.concat([cell_output]+attns, -1), int(output_size))
				if loop_function is not None:
					prev = output
				outputs.append(output)

	return (outputs, state)	


def embedding_attention_decoder(
	decoder_inputs,
	initial_state,
	attention_states,
	cell,
	num_symbols,
	embedding_size,
	num_heads=1,
	output_size=None,
	output_projection=None,
	feed_previous=False,
	update_embedding_for_previous=True,
	dtype=None,
	not_shared=False,
	scope=None,
	initial_state_attention=False):
	"""RNN decoder with embedding and attention and a pure-decoding option.

	Args:
		decoder_inputs: A list of 1D batch-sized int32 Tensors (decoder inputs).
		initial_state: 2D Tensor [batch_size x cell.state_size].
		attention_states: 3D Tensor [batch_size x attn_length x attn_size].
		cell: core_rnn_cell.RNNCell defining the cell function.
		num_symbols: Integer, how many symbols come into the embedding.
		embedding_size: Integer, the length of the embedding vector for each symbol.
		num_heads: Number of attention heads that read from attention_states.
		output_size: Size of the output vectors; if None, use output_size.
		output_projection: None or a pair (W, B) of output projection weights and
			biases; W has shape [output_size x num_symbols] and B has shape
			[num_symbols]; if provided and feed_previous=True, each fed previous
			output will first be multiplied by W and added B.
		feed_previous: Boolean; if True, only the first of decoder_inputs will be
			used (the "GO" symbol), and all other decoder inputs will be generated by:
				next = embedding_lookup(embedding, argmax(previous_output)),
			In effect, this implements a greedy decoder. It can also be used
			during training to emulate http://arxiv.org/abs/1506.03099.
			If False, decoder_inputs are used as given (the standard decoder case).
		update_embedding_for_previous: Boolean; if False and feed_previous=True,
			only the embedding for the first symbol of decoder_inputs (the "GO"
			symbol) will be updated by back propagation. Embeddings for the symbols
			generated from the decoder itself remain unchanged. This parameter has
			no effect if feed_previous=False.
		dtype: The dtype to use for the RNN initial states (default: tf.float32).
		scope: VariableScope for the created subgraph; defaults to
			"embedding_attention_decoder".
		initial_state_attention: If False (default), initial attentions are zero.
			If True, initialize the attentions from the initial state and attention
			states -- useful when we wish to resume decoding from a previously
			stored decoder state and attention states.

	Returns:
		A tuple of the form (outputs, state), where:
			outputs: A list of the same length as decoder_inputs of 2D Tensors with
				shape [batch_size x output_size] containing the generated outputs.
			state: The state of each decoder cell at the final time-step.
				It is a 2D Tensor of shape [batch_size x cell.state_size].

	Raises:
		ValueError: When output_projection has the wrong shape.
	"""
	if output_size is None:
		output_size = cell.output_size
	if output_projection is not None:
		proj_biases = ops.convert_to_tensor(output_projection[1], dtype=dtype)
		proj_biases.get_shape().assert_is_compatible_with([num_symbols])

	with variable_scope.variable_scope(
			scope or "embedding_attention_decoder", dtype=dtype) as scope:

		embedding = variable_scope.get_variable("embedding",
																						[num_symbols, embedding_size])
		loop_function = _extract_argmax_and_embed(
				embedding, output_projection,
				update_embedding_for_previous) if feed_previous else None
		emb_inp = [
				embedding_ops.embedding_lookup(embedding, i) for i in decoder_inputs
		]
		return attention_decoder(
				emb_inp,
				initial_state,
				attention_states,
				cell,
				output_size=output_size,
				num_heads=num_heads,
				loop_function=loop_function,
				not_shared=not_shared,
				initial_state_attention=initial_state_attention)


def embedding_attention_seq2seq(
	encoder_inputs,
	decoder_inputs,
	cell,
	num_encoder_symbols,
	num_decoder_symbols,
	embedding_size,
	num_heads=1,
	output_projection=None,
	feed_previous=False,
	dtype=None,
	not_shared=False,
	scope=None,
	initial_state_attention=False):
	"""Embedding sequence-to-sequence model with attention.

	This model first embeds encoder_inputs by a newly created embedding (of shape
	[num_encoder_symbols x input_size]). Then it runs an RNN to encode
	embedded encoder_inputs into a state vector. It keeps the outputs of this
	RNN at every step to use for attention later. Next, it embeds decoder_inputs
	by another newly created embedding (of shape [num_decoder_symbols x
	input_size]). Then it runs attention decoder, initialized with the last
	encoder state, on embedded decoder_inputs and attending to encoder outputs.

	Warning: when output_projection is None, the size of the attention vectors
	and variables will be made proportional to num_decoder_symbols, can be large.

	Args:
		encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
		decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
		cell: core_rnn_cell.RNNCell defining the cell function and size.
		num_encoder_symbols: Integer; number of symbols on the encoder side.
		num_decoder_symbols: Integer; number of symbols on the decoder side.
		embedding_size: Integer, the length of the embedding vector for each symbol.
		num_heads: Number of attention heads that read from attention_states.
		output_projection: None or a pair (W, B) of output projection weights and
			biases; W has shape [output_size x num_decoder_symbols] and B has
			shape [num_decoder_symbols]; if provided and feed_previous=True, each
			fed previous output will first be multiplied by W and added B.
		feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
			of decoder_inputs will be used (the "GO" symbol), and all other decoder
			inputs will be taken from previous outputs (as in embedding_rnn_decoder).
			If False, decoder_inputs are used as given (the standard decoder case).
		dtype: The dtype of the initial RNN state (default: tf.float32).
		scope: VariableScope for the created subgraph; defaults to
			"embedding_attention_seq2seq".
		initial_state_attention: If False (default), initial attentions are zero.
			If True, initialize the attentions from the initial state and attention
			states.

	Returns:
		A tuple of the form (outputs, state), where:
			outputs: A list of the same length as decoder_inputs of 2D Tensors with
				shape [batch_size x num_decoder_symbols] containing the generated
				outputs.
			state: The state of each decoder cell at the final time-step.
				It is a 2D Tensor of shape [batch_size x cell.state_size].
	"""
	with variable_scope.variable_scope(
			scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
		dtype = scope.dtype
		# Encoder.
		encoder_cell = EmbeddingWrapper(
				cell,
				embedding_classes=num_encoder_symbols,
				embedding_size=embedding_size)
		encoder_outputs, encoder_state = _rnn(
				encoder_cell, encoder_inputs, dtype=dtype, not_shared=not_shared)
		# the state return by rnn is a tuple includes some other stuff
		if isinstance(encoder_state, list):
			encoder_state = encoder_state[0]

		# First calculate a concatenation of encoder outputs to put attention on.
		top_states = [
				array_ops.reshape(e, [-1, 1, cell.output_size]) for e in encoder_outputs
		]
		attention_states = array_ops.concat(top_states,1)

		# Decoder.
		output_size = None
		if output_projection is None:
			cell = OutputProjectionWrapper(cell, num_decoder_symbols)
			output_size = num_decoder_symbols

		if isinstance(feed_previous, bool):
			return embedding_attention_decoder(
					decoder_inputs,
					encoder_state,
					attention_states,
					cell,
					num_decoder_symbols,
					embedding_size,
					num_heads=num_heads,
					output_size=output_size,
					output_projection=output_projection,
					feed_previous=feed_previous,
					not_shared=not_shared,
					initial_state_attention=initial_state_attention)

		# If feed_previous is a Tensor, we construct 2 graphs and use cond.
		def decoder(feed_previous_bool):
			reuse = None if feed_previous_bool else True
			with variable_scope.variable_scope(
					variable_scope.get_variable_scope(), reuse=reuse) as scope:
				outputs, state = embedding_attention_decoder(
						decoder_inputs,
						encoder_state,
						attention_states,
						cell,
						num_decoder_symbols,
						embedding_size,
						num_heads=num_heads,
						output_size=output_size,
						output_projection=output_projection,
						feed_previous=feed_previous_bool,
						update_embedding_for_previous=False,
						not_shared=not_shared,
						initial_state_attention=initial_state_attention)
				state_list = [state]
				if nest.is_sequence(state):
					state_list = nest.flatten(state)
				return outputs + state_list

		outputs_and_state = control_flow_ops.cond(feed_previous,
																							lambda: decoder(True),
																							lambda: decoder(False))
		outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
		state_list = outputs_and_state[outputs_len:]
		state = state_list[0]
		if nest.is_sequence(encoder_state):
			state = nest.pack_sequence_as(
					structure=encoder_state, flat_sequence=state_list)
		return outputs_and_state[:outputs_len], state