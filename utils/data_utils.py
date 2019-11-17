import json
import re
import random
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile

from . common_utils import *
import utils.PTB_data_utils as PTB_data_utils

Data_Mode = ['PTB_data']

def prepare_data(data_mode, inputs):
	if data_mode == 'PTB_data':
		data_path, vocab_path = inputs
		return PTB_data_utils.PTB_prepare_data(data_path, vocab_path)
	else:
		NotImplementedError
