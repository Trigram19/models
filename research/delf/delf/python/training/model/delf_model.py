# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""DELF model implementation based on the following paper.

  Large-Scale Image Retrieval with Attentive Deep Local Features
  https://arxiv.org/abs/1612.06321
"""

import tensorflow as tf

from delf.python.training.model import resnet50 as resnet

layers = tf.keras.layers
reg = tf.keras.regularizers

_DECAY = 0.0001

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):


class AttentionModel(tf.keras.Model):
  """Instantiates attention model.

  Uses two [kernel_size x kernel_size] convolutions and softplus as activation
  to compute an attention map with the same resolution as the featuremap.
  Features l2-normalized and aggregated using attention probabilites as weights.
  """

  def __init__(self, kernel_size=1, decay=_DECAY, name='attention'):
    """Initialization of attention model.

    Args:
      kernel_size: int, kernel size of convolutions.
      decay: float, decay for l2 regularization of kernel weights.
      name: str, name to identify model.
    """
    super(AttentionModel, self).__init__(name=name)

    # First convolutional layer (called with relu activation).
    self.conv1 = layers.Conv2D(
        512,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv1')
    self.bn_conv1 = layers.BatchNormalization(axis=3, name='bn_conv1')

    self.attn = Attention(512)
    # Second convolutional layer, with softplus activation.
    self.conv2 = layers.Conv2D(
        1,
        kernel_size,
        kernel_regularizer=reg.l2(decay),
        padding='same',
        name='attn_conv2')
    self.activation_layer = layers.Activation('softplus')

  def call(self, inputs, training=True):
    x = self.conv1(inputs)
    x = self.bn_conv1(x, training=training)
    x = self.attn(x)
    x = tf.nn.relu(x)

    score = self.conv2(x)
    prob = self.activation_layer(score)

    # L2-normalize the featuremap before pooling.
    inputs = tf.nn.l2_normalize(inputs, axis=-1)
    feat = tf.reduce_mean(tf.multiply(inputs, prob), [1, 2], keepdims=False)

    return feat, prob, score


class Delf(tf.keras.Model):
  """Instantiates Keras DELF model using ResNet50 as backbone.

  This class implements the [DELF](https://arxiv.org/abs/1612.06321) model for
  extracting local features from images. The backbone is a ResNet50 network
  that extracts featuremaps from both conv_4 and conv_5 layers. Activations
  from conv_4 are used to compute an attention map of the same resolution.
  """

  def __init__(self, block3_strides=True, name='DELF'):
    """Initialization of DELF model.

    Args:
      block3_strides: bool, whether to add strides to the output of block3.
      name: str, name to identify model.
    """
    super(Delf, self).__init__(name=name)

    # Backbone using Keras ResNet50.
    self.backbone = resnet.ResNet50(
        'channels_last',
        name='backbone',
        include_top=False,
        pooling='avg',
        block3_strides=block3_strides,
        average_pooling=False)

    # Attention model.
    self.attention = AttentionModel(name='attention')

  # Define classifiers for training backbone and attention models.
  def init_classifiers(self, num_classes):
    self.num_classes = num_classes
    self.desc_classification = layers.Dense(
        num_classes, activation=None, kernel_regularizer=None, name='desc_fc')

    self.attn_classification = layers.Dense(
        num_classes, activation=None, kernel_regularizer=None, name='att_fc')

  # Weights to optimize for descriptor fine tuning.
  @property
  def desc_trainable_weights(self):
    return (self.backbone.trainable_weights +
            self.desc_classification.trainable_weights)

  # Weights to optimize for attention model training.
  @property
  def attn_trainable_weights(self):
    return (self.attention.trainable_weights +
            self.attn_classification.trainable_weights)

  def call(self, input_image, training=True):
    blocks = {}

    self.backbone.build_call(
        input_image, intermediates_dict=blocks, training=training)

    features = blocks['block3']  # pytype: disable=key-error
    _, probs, _ = self.attention(features, training=training)

    return probs, features
