"""
Written by TH, Kim
This code is Siamese Pseudo Invariant Feature Network (SIPIF-net).
The SIPIF-net is designed to extract the PIFs for relative radiometric normalization
"""
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from Models.Attention_module import *
from tensorflow.keras.applications import VGG16
from tensorflow.keras import backend as K


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sumSquared = K.sum(K.square(featsA - featsB), axis=1,
                       keepdims=True)
    # return the euclidean distance between the vectors
    return K.sqrt(K.maximum(sumSquared, K.epsilon()))


def SIPIF_net(inputShape1, inputShape2):
    inputs_ref = Input(inputShape1)  # Reference_input layer
    inputs_sen = Input(inputShape2)  # Sensed_input layer

    # Stage 1 + CBAM
    st1_conv1_ref = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs_ref)
    st1_bn1_ref = BatchNormalization()(st1_conv1_ref)
    st1_conv2_ref = Conv2D(64,3 ,padding='same', kernel_initializer='he_normal')(st1_bn1_ref)
    st1_bn2_ref = BatchNormalization()(st1_conv2_ref)
    st1_pool_ref = MaxPooling2D(pool_size=(2,2))(st1_bn2_ref)
    st1_CH_ref = ChannelAttention(256, 8)(st1_pool_ref)
    st1_SP_ref = SpatialAttention(3)(st1_CH_ref)

    st1_conv1_sen = Conv2D(64, 3, padding='same', kernel_initializer='he_normal')(inputs_sen)
    st1_bn1_sen = BatchNormalization()(st1_conv1_sen)
    st1_conv2_sen = Conv2D(64,3 ,padding='same', kernel_initializer='he_normal')(st1_bn1_sen)
    st1_bn2_sen = BatchNormalization()(st1_conv2_sen)
    st1_pool_sen = MaxPooling2D(pool_size=(2,2))(st1_bn2_sen)
    st1_CH_sen = ChannelAttention(256, 8)(st1_pool_sen)
    st1_SP_sen = SpatialAttention(3)(st1_CH_sen)

    # Stage 2 + CBAM
    st2_conv1_ref = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(st1_SP_ref)
    st2_bn1_ref = BatchNormalization()(st2_conv1_ref)
    st2_conv2_ref = Conv2D(128,3 ,padding='same', kernel_initializer='he_normal')(st2_bn1_ref)
    st2_bn2_ref = BatchNormalization()(st2_conv2_ref)
    st2_pool_ref = MaxPooling2D(pool_size=(2,2))(st2_bn2_ref)
    st2_CH_ref = ChannelAttention(256, 8)(st2_pool_ref)
    st2_SP_ref = SpatialAttention(3)(st2_CH_ref)

    st2_conv1_sen = Conv2D(128, 3, padding='same', kernel_initializer='he_normal')(st1_SP_sen)
    st2_bn1_sen = BatchNormalization()(st2_conv1_sen)
    st2_conv2_sen = Conv2D(128,3 ,padding='same', kernel_initializer='he_normal')(st2_bn1_sen)
    st2_bn2_sen = BatchNormalization()(st2_conv2_sen)
    st2_pool_sen = MaxPooling2D(pool_size=(2,2))(st2_bn2_sen)
    st2_CH_sen = ChannelAttention(256, 8)(st2_pool_sen)
    st2_SP_sen = SpatialAttention(3)(st2_CH_sen)


    # Stage 3 + CBAM
    st3_conv1_ref = Conv2D(256,3, padding='valid', kernel_initializer='he_normal')(st2_SP_ref)
    st3_bn1_ref = BatchNormalization()(st3_conv1_ref)
    st3_pool_ref = MaxPooling2D(pool_size=(2,2))(st3_bn1_ref)
    st3_CH_ref = ChannelAttention(256, 8)(st3_pool_ref)
    st3_SP_ref = SpatialAttention(3)(st3_CH_ref)

    st3_conv1_sen = Conv2D(256,3, padding='valid', kernel_initializer='he_normal')(st2_SP_sen)
    st3_bn1_sen = BatchNormalization()(st3_conv1_sen)
    st3_pool_sen = MaxPooling2D(pool_size=(2,2))(st3_bn1_sen)
    st3_CH_sen = ChannelAttention(256, 8)(st3_pool_sen)
    st3_SP_sen = SpatialAttention(3)(st3_CH_sen)

    # Stage 4 + CBAM
    st4_conv1_ref = Conv2D(512,3, padding='valid', kernel_initializer='he_normal')(st3_SP_ref)
    st4_bn1_ref = BatchNormalization()(st4_conv1_ref)
    st4_pool_ref = MaxPooling2D(pool_size=(2,2))(st4_bn1_ref)
    st4_CH_ref = ChannelAttention(512, 8)(st4_pool_ref)
    st4_SP_ref = SpatialAttention(3)(st4_CH_ref)
    outputs_ref = Flatten()(st4_SP_ref)

    st4_conv1_sen = Conv2D(512, 3, padding='valid', kernel_initializer='he_normal')(st3_SP_sen)
    st4_bn1_sen = BatchNormalization()(st4_conv1_sen)
    st4_pool_sen = MaxPooling2D(pool_size=(2,2))(st4_bn1_sen)
    st4_CH_sen = ChannelAttention(512, 8)(st4_pool_sen)
    st4_SP_sen = SpatialAttention(3)(st4_CH_sen)
    outputs_sen = Flatten()(st4_SP_sen)

    # Estimation of similarity between reference and sensed deep features
    L2_distance = Lambda(euclidean_distance)([outputs_ref, outputs_sen])
    outputs = Dense(1, activation="sigmoid")(L2_distance)

    # Build the SIPIF_net
    SIPIF_net = Model(inputs=[inputs_ref,inputs_sen], outputs= outputs)
    SIPIF_net.summary()

    return SIPIF_net





