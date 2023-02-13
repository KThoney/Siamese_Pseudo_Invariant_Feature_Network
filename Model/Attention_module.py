"""
Written by TH, Kim
This code is convolutional attention module.
"""
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.utils import plot_model


class ChannelAttention(Layer):
    def __init__(self, filters, ratio):
        super(ChannelAttention, self).__init__()
        self.filters = filters
        self.ratio = ratio

        def build(self, input_shape):
            self.shared_layer_one = Dense(self.filters // self.ratio, activation='relu', kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')
            self.shared_layer_two = Dense(self.filters, kernel_initializer='he_normal',
                                          use_bias=True,
                                          bias_initializer='zeros')

        def call(self, inputs):
            # AvgPool
            avg_pool = GlobalAveragePooling2D()(inputs)

            avg_pool = self.shared_layer_one(avg_pool)
            avg_pool = self.shared_layer_two(avg_pool)

            # MaxPool
            max_pool = GlobalMaxPooling2D()(inputs)
            max_pool = Reshape((1, 1, filters))(max_pool)

            max_pool = self.shared_layer_one(max_pool)
            max_pool = self.shared_layer_two(max_pool)

            attention = Add()([avg_pool, max_pool])
            attention = Activation('sigmoid')(attention)

            return Multiply()([inputs, attention])


class SpatialAttention(Layer):
    def __init__(self, kernel_size):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size

        def build(self, input_shape):
            self.conv2d = Conv2D(filters=1, kernel_size=self.kernel_size,
                                 strides=1,
                                 padding='same',
                                 activation='sigmoid',
                                 kernel_initializer='he_normal',
                                 use_bias=False)

        def call(self, inputs):
            # AvgPool
            avg_pool = Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(inputs)

            # MaxPool
            max_pool = Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(inputs)

            attention = Concatenate(axis=3)([avg_pool, max_pool])

            attention = self.conv2d(attention)

            return Multiply([inputs, attention])
