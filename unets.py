ROWS = 256
COLS = 256
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Conv3D, Lambda, Permute, Concatenate, multiply
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Conv3D, GlobalAveragePooling3D, GlobalMaxPooling3D, Reshape, Permute, Lambda, Activation, Multiply, Add

def eca_block3(input_feature, k_size=3):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    channel = input_feature.shape[channel_axis]

    shared_layer_one = Conv3D(filters=1, kernel_size=(k_size, 1, 1), strides=1, kernel_initializer='he_normal', use_bias=False, padding="same")

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = Reshape((1, 1, 1, channel))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)
    avg_pool = Permute((4, 1, 2, 3))(avg_pool)
    # avg_pool = Lambda(squeeze)(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    # avg_pool = Lambda(unsqueeze)(avg_pool)
    avg_pool = Permute((2, 3, 4, 1))(avg_pool)
    assert avg_pool.shape[1:] == (1, 1, 1, channel)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = Reshape((1, 1, 1, channel))(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)
    max_pool = Permute((4, 1, 2, 3))(max_pool)
    # max_pool = Lambda(squeeze)(max_pool)
    max_pool = shared_layer_one(max_pool)
    # max_pool = Lambda(unsqueeze)(max_pool)
    max_pool = Permute((2, 3, 4, 1))(max_pool)
    assert max_pool.shape[1:] == (1, 1, 1, channel)

    eca_feature = Add()([avg_pool, max_pool])
    eca_feature = Activation('sigmoid')(eca_feature)

    if K.image_data_format() == "channels_first":
        eca_feature = Permute((4, 1, 2, 3))(eca_feature)

    return Multiply()([input_feature, eca_feature])

def unsqueeze(input):
    return K.expand_dims(input, axis=-1)

def squeeze(input):
    return K.squeeze(input, axis=-1)
def spatial_attention(input_feature):
    kernel_size = (14,14, 1)

    if K.image_data_format() == "channels_first":
        channel = input_feature.shape[1]
        sa_feature = Permute((2, 3, 4, 1))(input_feature)
    else:
        channel = input_feature.shape[-1]
        sa_feature = input_feature

    avg_pool = Lambda(lambda x: K.mean(x, axis=4, keepdims=True))(sa_feature)
    assert avg_pool.shape[-1] == 1
    max_pool = Lambda(lambda x: K.max(x, axis=4, keepdims=True))(sa_feature)
    assert max_pool.shape[-1] == 1
    concat = Concatenate(axis=4)([avg_pool, max_pool])
    assert concat.shape[-1] == 2
    sa_feature = Conv3D(filters=1,
                          kernel_size=kernel_size,
                          strides=1,
                          padding='same',
                          activation='sigmoid',
                          kernel_initializer='he_normal',
                          use_bias=False)(concat)
    assert sa_feature.shape[-1] == 1

    if K.image_data_format() == "channels_first":
        sa_feature = Permute((4, 1, 2, 3))(sa_feature)

    return multiply([input_feature, sa_feature])


def ASPP_3D(x, filter):
    shape = x.shape

    y1 = AveragePooling3D(pool_size=(shape[1], shape[2], shape[3]))(x)
    y1 = Conv3D(filter, 1, padding="same", kernel_initializer='he_normal')(y1)
    y1 = GroupNormalization(groups=8)(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling3D((shape[1], shape[2], shape[3]))(y1)

    y2 = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y2 = GroupNormalization(groups=8)(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv3D(filter, 3, dilation_rate=6, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y3 = GroupNormalization(groups=8)(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv3D(filter, 3, dilation_rate=12, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y4 = GroupNormalization(groups=8)(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv3D(filter, 3, dilation_rate=16, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y5 = GroupNormalization(groups=8)(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(y)
    y = GroupNormalization(groups=8)(y)
    y = Activation("relu")(y)

    return y



def attention_block(F_g, F_l, F_int):
    g = Conv3D(F_int, 1, padding="valid",kernel_initializer='he_normal')(F_g)
    g =BatchNormalization()(g)

    x = Conv3D(F_int, 1, padding="valid",kernel_initializer='he_normal')(F_l)
    x = BatchNormalization()(x)

    psi = Add()([g, x])
    psi = Activation("relu")(psi)

    psi = Conv3D(1, 1, padding="valid",kernel_initializer='he_normal')(psi)
    psi = BatchNormalization()(psi)
    psi = Activation("sigmoid")(psi)

    return Multiply()([F_l, psi])
    


def att_aspp(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)


    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = UpSampling3D((2, 2, 1))(convm)
    conv5 = attention_block(up5, conv3, n_channel*4)
    up5 = concatenate([up5,conv5],axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = UpSampling3D((2, 2, 1))(conv5)
    conv6 = attention_block(up6, conv2, n_channel*2)
    up6 = concatenate([up6,conv6],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = UpSampling3D((2, 2, 1))(conv6)
    conv7 = attention_block(up7, conv1, n_channel)
    up7 = concatenate([up7,conv7],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model



def backbonesasapp(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)


    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 2))(convm),spatial_attention(conv3)],axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),spatial_attention(conv2)],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), spatial_attention(conv1)],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model





def ASPP_3D2(x, filter):
    shape = x.shape

    y1 = AveragePooling3D(pool_size=(shape[1], shape[2], shape[3]))(x)
    y1 = Conv3D(filter, 1, padding="same", kernel_initializer='he_normal')(y1)
    y1 = GroupNormalization(groups=8)(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling3D((shape[1], shape[2], shape[3]))(y1)

    y2 = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y2 = GroupNormalization(groups=8)(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv3D(filter, 3, dilation_rate=4, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y3 = GroupNormalization(groups=8)(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv3D(filter, 3, dilation_rate=8, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y4 = GroupNormalization(groups=8)(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv3D(filter, 3, dilation_rate=12, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y5 = GroupNormalization(groups=8)(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(y)
    y = GroupNormalization(groups=8)(y)
    y = Activation("relu")(y)

    return y

def ASPP_3D3(x, filter):
    shape = x.shape

    y1 = AveragePooling3D(pool_size=(shape[1], shape[2], shape[3]))(x)
    y1 = Conv3D(filter, 1, padding="same", kernel_initializer='he_normal')(y1)
    y1 = GroupNormalization(groups=8)(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling3D((shape[1], shape[2], shape[3]))(y1)

    y2 = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y2 = GroupNormalization(groups=8)(y2)
    y2 = Activation("relu")(y2)

    y3 = Conv3D(filter, 3, dilation_rate=6, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y3 = GroupNormalization(groups=8)(y3)
    y3 = Activation("relu")(y3)

    y4 = Conv3D(filter, 3, dilation_rate=12, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y4 = GroupNormalization(groups=8)(y4)
    y4 = Activation("relu")(y4)

    y5 = Conv3D(filter, 3, dilation_rate=18, padding="same", use_bias=False, kernel_initializer='he_normal')(x)
    y5 = GroupNormalization(groups=8)(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])
    y = Conv3D(filter, 1, dilation_rate=1, padding="same", use_bias=False, kernel_initializer='he_normal')(y)
    y = GroupNormalization(groups=8)(y)
    y = Activation("relu")(y)

    return y

def saunetwaspp(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)


    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D2(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 2))(convm),spatial_attention(conv3)],axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),spatial_attention(conv2)],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), spatial_attention(conv1)],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model

def saunetwaspp3(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)


    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D3(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 2))(convm),spatial_attention(conv3)],axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),spatial_attention(conv2)],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), spatial_attention(conv1)],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model








def backbonesasapp2(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)


    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 1))(convm),spatial_attention(conv3)],axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),spatial_attention(conv2)],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), spatial_attention(conv1)],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model








def backbone():
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(24, (3, 3, 3), padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    conv1 = Conv3D(24, (3, 3, 3), padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)


    conv2 = Conv3D(48, (3, 3, 3),  padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(48, (3, 3, 3), padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(96, (3, 3, 3), padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(96, (3, 3, 3), padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(192, (3, 3, 3), padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Conv3D(192, (3, 3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    # expansive/synthesis path

    up5 = concatenate([UpSampling3D((2, 2, 2))(conv4), conv3], axis=4)
    conv5 = Conv3D(96, (3, 3, 3), padding='same',kernel_initializer='he_normal')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(96, (3, 3, 3), padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 2))(conv5), conv2], axis=4)
    conv6 = Conv3D(48, (3, 3, 3),padding='same',kernel_initializer='he_normal')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(48, (3, 3, 3), padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 2))(conv6), conv1], axis=4)
    conv7 = Conv3D(24, (3, 3, 3), padding='same',kernel_initializer='he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(24, (3, 3, 3),  padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model
def backbones():
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(24, (3, 3, 1), padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    conv1 = Conv3D(24, (3, 3, 1), padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)


    conv2 = Conv3D(48, (3, 3, 1),  padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(48, (3, 3, 1), padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(96, (3, 3, 1), padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(96, (3, 3, 1), padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)

    conv4 = Conv3D(192, (3, 3, 3), padding='same',kernel_initializer='he_normal')(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    conv4 = Conv3D(192, (3, 3, 3), activation='relu', padding='same',kernel_initializer='he_normal')(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)
    # expansive/synthesis path

    up5 = concatenate([UpSampling3D((2, 2, 1))(conv4), conv3], axis=4)
    conv5 = Conv3D(96, (3, 3, 1), padding='same',kernel_initializer='he_normal')(up5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(96, (3, 3, 1), padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5), conv2], axis=4)
    conv6 = Conv3D(48, (3, 3, 1),padding='same',kernel_initializer='he_normal')(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(48, (3, 3, 1), padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), conv1], axis=4)
    conv7 = Conv3D(24, (3, 3, 1), padding='same',kernel_initializer='he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(24, (3, 3, 1),  padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model


def backbonewgn(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)



    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 1))(convm), conv3],
                      axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),conv2],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), conv1],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model






def saunet(n_channel=24):
    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)

    conv2 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)



    convm = Conv3D(n_channel*8, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = Conv3D(n_channel*8, (3, 3, 13), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)

    up5 = concatenate([UpSampling3D((2, 2, 1))(convm), spatial_attention(conv3)],
                      axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5),spatial_attention(conv2)],axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 3), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6),spatial_attention(conv1)],axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)

    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model







def 3DSAUNet(n_channel=24):

    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    

    conv2 = Conv3D(n_channel* 2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*  2, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)



    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same', kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = spatial_attention(convm)
    convm = Conv3D(n_channel*8, (3, 3, 3), activation=None, padding='same', kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)



    up5 = concatenate([UpSampling3D((2, 2, 1))(convm), spatial_attention(conv3)],
                      axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3,1), activation=None, padding='same', kernel_initializer='he_normal')(conv5)

    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5), spatial_attention(conv2)],
                      axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), spatial_attention(conv1)],
                      axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same', kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax', kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model


def aspp_unet(n_channel=24):

    inputs = Input((ROWS, COLS, 32, 1))
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(inputs)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    conv1 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv1)
    conv1 = GroupNormalization(groups=8)(conv1)
    conv1 = Activation("relu")(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 1))(conv1)

    conv2 = Conv3D(n_channel* 2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool1)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    conv2 = Conv3D(n_channel*  2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv2)
    conv2 = GroupNormalization(groups=8)(conv2)
    conv2 = Activation("relu")(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 1))(conv2)

    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool2)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    conv3 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv3)
    conv3 = GroupNormalization(groups=8)(conv3)
    conv3 = Activation("relu")(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 1))(conv3)



    convm = Conv3D(n_channel*8, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(pool3)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)
    convm = ASPP_3D(convm,n_channel*8)
    convm = Conv3D(n_channel*8, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(convm)
    convm = GroupNormalization(groups=8)(convm)
    convm = Activation("relu")(convm)



    up5 = concatenate([UpSampling3D((2, 2, 1))(convm), (conv3)],
                      axis=4)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)
    conv5 = Conv3D(n_channel*4, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv5)
    conv5 = GroupNormalization(groups=8)(conv5)
    conv5 = Activation("relu")(conv5)

    up6 = concatenate([UpSampling3D((2, 2, 1))(conv5), (conv2)],
                      axis=4)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)
    conv6 = Conv3D(n_channel*2, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv6)
    conv6 = GroupNormalization(groups=8)(conv6)
    conv6 = Activation("relu")(conv6)

    up7 = concatenate([UpSampling3D((2, 2, 1))(conv6), (conv1)],
                      axis=4)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(up7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)
    conv7 = Conv3D(n_channel, (3, 3, 1), activation=None, padding='same',kernel_initializer='he_normal')(conv7)
    conv7 = GroupNormalization(groups=8)(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv3D(3, (1, 1, 1), activation='softmax',kernel_initializer='he_normal')(conv7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model



