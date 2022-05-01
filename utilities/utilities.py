
import datetime
import time
import os
import pickle as pkl

start_time = time.time()


def get_uptime():
    return '{}'.format(datetime.timedelta(seconds=time.time() - start_time))


def list_files_in_dir(path: str):
    li = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    li.sort()
    return li


def list_file_paths_in_dir(path: str):
    li = [os.path.join(path, f) for f in os.listdir(
        path) if os.path.isfile(os.path.join(path, f))]
    li.sort()
    return li

def save_similarity_vector(vector, path):
    with open(path, "wb") as f:
        pkl.dump(vector, f)

def load_similarity_vector(path):
    with open(path, "rb") as f:
        return pkl.load(f)

# Block adding
# Args of last 

import collections

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'strides', 'se_ratio', 'conv_type'
])

last = BlockArgs(kernel_size=3, num_repeat=5, input_filters=304, output_filters=512,
            expand_ratio=6, id_skip=True, strides=[1, 1], se_ratio=0.25, conv_type=0)

new = BlockArgs(kernel_size=3, num_repeat=5, input_filters=512, output_filters=512,
            expand_ratio=6, id_skip=True, strides=[2, 2], se_ratio=0.25, conv_type=0)

import string

from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import utils as keras_utils

#from .utils import CONV_KERNEL_INITIALIZER
#from .utils import DENSE_KERNEL_INITIALIZER
#from .utils import round_filters
#from .utils import round_repeats
#from .config import *


def get_dropout():
    """Wrapper over custom dropout. Fix problem of ``None`` shape for tf.keras.
    It is not possible to define FixedDropout class as global object,
    because we do not have modules for inheritance at first time.
    Issue:
        https://github.com/tensorflow/tensorflow/issues/30946
    """

    class FixedDropout(layers.Dropout):
        def _get_noise_shape(self, inputs):
            if self.noise_shape is None:
                return self.noise_shape

            symbolic_shape = backend.shape(inputs)
            noise_shape = [symbolic_shape[axis] if shape is None else shape
                        for axis, shape in enumerate(self.noise_shape)]
            return tuple(noise_shape)

    return FixedDropout


def mb_conv_block(inputs,
                block_args: BlockArgs,
                activation='swish',
                drop_rate=None,
                prefix='',
                conv_dropout=None,
                mb_type='normal'):
    """Fused Mobile Inverted Residual Bottleneck"""
    assert mb_type in ['normal', 'fused']
    has_se = (block_args.se_ratio is not None) and (0 < block_args.se_ratio <= 1)
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    Dropout = get_dropout()

    x = inputs

    # Expansion phase
    filters = block_args.input_filters * block_args.expand_ratio
    if block_args.expand_ratio != 1:
        x = layers.Conv2D(filters,
                        1 if mb_type == 'normal' else block_args.kernel_size,
                        strides=1 if mb_type == 'normal' else block_args.strides,
                        kernel_initializer=CONV_KERNEL_INITIALIZER,
                        padding='same',
                        use_bias=False,
                        name=f'{prefix}expand_conv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}expand_bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}expand_activation')(x)

    if mb_type is 'normal':
        x = layers.DepthwiseConv2D(block_args.kernel_size,
                                block_args.strides,
                                depthwise_initializer=CONV_KERNEL_INITIALIZER,
                                padding='same',
                                use_bias=False,
                                name=f'{prefix}dwconv')(x)
        x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}bn')(x)
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if conv_dropout and block_args.expand_ratio > 1:
        x = layers.Dropout(conv_dropout)(x)

    if has_se:
        num_reduced_filters = max(1, int(
            block_args.input_filters * block_args.se_ratio
        ))
        se_tensor = layers.GlobalAveragePooling2D(name=prefix + 'se_squeeze')(x)

        target_shape = (1, 1, filters) if backend.image_data_format() == 'channels_last' else (filters, 1, 1)
        se_tensor = layers.Reshape(target_shape, name=prefix + 'se_reshape')(se_tensor)
        se_tensor = layers.Conv2D(num_reduced_filters, 1,
                                activation=activation,
                                padding='same',
                                use_bias=True,
                                kernel_initializer=CONV_KERNEL_INITIALIZER,
                                name=prefix + 'se_reduce')(se_tensor)
        se_tensor = layers.Conv2D(filters, 1,
                                activation='sigmoid',
                                padding='same',
                                use_bias=True,
                                kernel_initializer=CONV_KERNEL_INITIALIZER,
                                name=prefix + 'se_expand')(se_tensor)
        x = layers.multiply([x, se_tensor], name=prefix + 'se_excite')

    # Output phase
    x = layers.Conv2D(block_args.output_filters,
                    kernel_size=1 if block_args.expand_ratio != 1 else block_args.kernel_size,
                    strides=1 if block_args.expand_ratio != 1 else block_args.strides,
                    kernel_initializer=CONV_KERNEL_INITIALIZER,
                    padding='same',
                    use_bias=False,
                    name=f'{prefix}project_conv')(x)

    x = layers.BatchNormalization(axis=bn_axis, name=f'{prefix}project_bn')(x)
    if block_args.expand_ratio == 1:
        x = layers.Activation(activation=activation, name=f'{prefix}activation')(x)

    if all(s == 1 for s in block_args.strides) \
            and block_args.input_filters == block_args.output_filters:
        if drop_rate and drop_rate > 0:
            x = Dropout(drop_rate,
                        noise_shape=(None, 1, 1, 1),
                        name=f'{prefix}dropout')(x)
        x = layers.Add(name=f'{prefix}add')([x, inputs])
    return x

CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        # EfficientNet actually uses an untruncated normal distribution for
        # initializing conv layers, but keras.initializers.VarianceScaling use
        # a truncated distribution.
        # We decided against a custom initializer for better serializability.
        'distribution': 'normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}

def round_filters(filters, width_coefficient, depth_divisor):
    """Round number of filters based on width multiplier."""

    filters *= width_coefficient
    new_filters = int(filters + depth_divisor / 2) // depth_divisor * depth_divisor
    new_filters = max(depth_divisor, new_filters)
    # Make sure that round down does not go down by more than 10%.
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)
