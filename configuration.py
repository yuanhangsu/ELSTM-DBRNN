# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ModelConfig(object):
  """Wrapper class for model hyperparameters."""

  def __init__(self):
    """Sets the default model hyperparameters."""

    # Number of unique words in the vocab (plus 1, for <UNK>).
    # The default value is larger than the expected actual vocab size to allow
    # for differences between tokenizer versions used in preprocessing. There is
    # no harm in using a value greater than the actual vocab size, but using a
    # value less than the actual vocab size will result in an error.
    self.num_input_symbols = 12000
    self.num_output_symbols = 12000

    # Batch size.
    self.batch_size = 50

    # Layers of RNN
    self.num_layers = 1

    # maximum input sequence length
    self.max_input_seq_length = 10
    self.max_output_seq_length = 10
    self.max_cell_length = 0

    # RNN Input and output dimensionality, respectively.
    self.embedding_size = 5
    self.cell_units = 5
    self.cell_mul = 2

    # RNN activation functions
    # Candidate: tanh, sigmoid, relu
    activation_func = "tanh"
    if activation_func == "tanh":
        self.activation_func = tf.tanh
    elif activation_func == "sigmoid":
        self.activation_func = tf.sigmoid
    elif activation_func == "relu":
        self.activation_func = tf.nn.relu
    elif activation_func == "relu6":
        self.activation_func = tf.nn.relu6

    # # of samples for sampled softmax, not use if 0
    self.num_samples = 0

    # maximum training steps
    self.max_training_steps = 0

    # Learning rate for the initial phase of training.
    self.learning_rate = 0.5
    self.learning_rate_decay_factor = 0.99

    # Clip gradients to this norm
    self.max_gradient_norm = 5.0

    # Checkpoint settings
    self.steps_per_checkpoint = 50
    self.max_checkpoints_to_keep = 5