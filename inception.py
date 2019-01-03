# Keras Core
from keras.layers import Conv2D, Dropout, Dense, Activation, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import Concatenate
from keras import backend as K


K.set_image_data_format('channels_first')

DEFAULT_CHANNEL = 'channels_first'

if DEFAULT_CHANNEL == 'channels_first':
    ch_axis = 1
else:
    ch_axis = -1

NB_OUTPUTS = 1

def conv2d_bn(x,
              filters,
              num_row,
              num_col,
              padding='same',
              strides=(1, 1),
              name=None):
    """Utility function to apply conv + BN.

    Modified from https://github.com/fchollet/keras/blob/master/keras/applications/inception_v3.py)

    # Arguments
        x: input tensor.
        filters: filters in `Conv2D`.
        num_row: height of the convolution kernel.
        num_col: width of the convolution kernel.
        padding: padding mode in `Conv2D`.
        strides: strides in `Conv2D`.
        channel: channel axes for BatchNormalization.
        name: name of the ops; will become `name + '_conv'`
            for the convolution.
    # Returns
        Output tensor after applying `Conv2D` and `BatchNormalization`.
    """
    if name is not None:
        conv_name = '{}_{}_{}x{}'.format('Conv2d_', name, num_row, num_col)
    else:
        conv_name = None

    x = Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        name=conv_name,
        use_bias=False)(x)
    x = BatchNormalization(axis=ch_axis, scale=False)(x)
    x = Activation('relu')(x)
    return x


def stem(input, prefix):

    x = conv2d_bn(input, 32, 3, 3, strides=(2, 2), padding='valid', name=prefix+'s1')
    x = conv2d_bn(x, 32, 3, 3, padding='valid', name=prefix+'s2')
    x = conv2d_bn(x, 64, 3, 3, name=prefix+'s3')

    b0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', data_format=DEFAULT_CHANNEL)(x)
    b1 = conv2d_bn(x, 96, 3, 3, strides=(2, 2), padding='valid', name=prefix+'s411')

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'s5')([b0, b1])

    b0 = conv2d_bn(x, 64, 1, 1, name=prefix+'s611')
    b0 = conv2d_bn(b0, 96, 3, 3, padding='valid', name=prefix+'s612')

    b1 = conv2d_bn(x, 64, 1, 1, name=prefix+'s621')
    b1 = conv2d_bn(b1, 64, 7, 1, name=prefix+'s622')
    b1 = conv2d_bn(b1, 64, 1, 7, name=prefix+'s623')
    b1 = conv2d_bn(b1, 96, 3, 3, padding='valid', name=prefix+'s624')

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'s7')([b0, b1])

    b0 = conv2d_bn(x, 196, 3, 3, strides=(2, 2), padding='valid', name=prefix+'s811')
    b1 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', data_format=DEFAULT_CHANNEL)(x)

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'s9')([b0, b1])

    return x


def inception_a(input, prefix):
    b0 = AveragePooling2D((3, 3), strides=1, padding='same', data_format=DEFAULT_CHANNEL)(input)
    b0 = conv2d_bn(b0, 96, 1, 1, name=prefix+'a112')

    b1 = conv2d_bn(input, 96, 1, 1, name=prefix+'a121')

    b2 = conv2d_bn(input, 64, 1, 1, name=prefix+'a131')
    b2 = conv2d_bn(b2, 96, 3, 3, name=prefix+'a132')

    b3 = conv2d_bn(input, 64, 1, 1, name=prefix+'a141')
    b3 = conv2d_bn(b3, 96, 3, 3, name=prefix+'a142')
    b3 = conv2d_bn(b3, 96, 3, 3, name=prefix+'a143')

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'a2')([b0, b1, b2, b3])
    return x


def inception_b(input, prefix):
    b0 = AveragePooling2D((3, 3), strides=1, padding='same', data_format=DEFAULT_CHANNEL)(input)
    b0 = conv2d_bn(b0, 128, 1, 1, name=prefix+'b112')

    b1 = conv2d_bn(input, 384, 1, 1, name=prefix+'b121')

    b2 = conv2d_bn(input, 192, 1, 1, name=prefix+'b131')
    b2 = conv2d_bn(b2, 224, 7, 1, name=prefix+'b132')
    b2 = conv2d_bn(b2, 256, 1, 7, name=prefix + 'b133')

    b3 = conv2d_bn(input, 192, 1, 1, name=prefix+'b141')
    b3 = conv2d_bn(b3, 192, 1, 7, name=prefix+'b142')
    b3 = conv2d_bn(b3, 224, 7, 1, name=prefix + 'b143')
    b3 = conv2d_bn(b3, 224, 1, 7, name=prefix+'b144')
    b3 = conv2d_bn(b3, 256, 7, 1, name=prefix + 'a145')

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'b2')([b0, b1, b2, b3])
    return x


def inception_c(input, prefix):
    b0 = AveragePooling2D((3, 3), strides=1, padding='same', data_format=DEFAULT_CHANNEL)(input)
    b0 = conv2d_bn(b0, 256, 1, 1, name=prefix+'c112')

    b1 = conv2d_bn(input, 256, 1, 1, name=prefix+'c121')

    b20 = conv2d_bn(input, 384, 1, 1, name=prefix+'c131')
    b21 = conv2d_bn(b20, 256, 1, 3, name=prefix + 'c1321')
    b22 = conv2d_bn(b20, 256, 3, 1, name=prefix+'c1322')

    b30 = conv2d_bn(input, 384, 1, 1, name=prefix+'c141')
    b30 = conv2d_bn(b30, 448, 1, 3, name=prefix+'c142')
    b30 = conv2d_bn(b30, 512, 3, 1, name=prefix + 'c143')
    b31 = conv2d_bn(b30, 256, 3, 1, name=prefix+'c1441')
    b32 = conv2d_bn(b30, 256, 1, 3, name=prefix + 'c1442')

    x = Concatenate(axis=ch_axis, name='concat_'+prefix+'c2')([b0, b1, b21, b22, b31, b32])

    return x

#k=192, l=224, m=256, n=384
def reduction_a(input, prefix):
    b0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', data_format=DEFAULT_CHANNEL)(input)

    b1 = conv2d_bn(input, 384, 3, 3, strides=(2, 2), padding='valid', name=prefix + 'ra121')

    b2 = conv2d_bn(input, 192, 1, 1, name=prefix + 'ra131')
    b2 = conv2d_bn(b2, 224, 3, 3, name=prefix + 'ra132')
    b2 = conv2d_bn(b2, 256, 3, 3, strides=(2, 2), padding='valid', name=prefix + 'ra133')

    x = Concatenate(axis=ch_axis, name='concat_' + prefix + 'a2')([b0, b1, b2])
    return x


def reduction_b(input, prefix):
    b0 = MaxPooling2D((3, 3), strides=(2, 2), padding='valid', data_format=DEFAULT_CHANNEL)(input)

    b1 = conv2d_bn(input, 192, 1, 1, name=prefix + 'rb121')
    b1 = conv2d_bn(b1, 192, 3, 3, strides=(2, 2), padding='valid', name=prefix + 'rb121')

    b2 = conv2d_bn(input, 256, 1, 1, name=prefix + 'rb131')
    b2 = conv2d_bn(b2, 256, 1, 7, name=prefix + 'rb132')
    b2 = conv2d_bn(b2, 256, 7, 1, name=prefix + 'rb133')
    b2 = conv2d_bn(b2, 256, 3, 3, strides=(2, 2), padding='valid', name=prefix + 'rb134')

    x = Concatenate(axis=ch_axis, name='concat_' + prefix + 'a2')([b0, b1, b2])
    return x


def build_inception_v4(input, enable_reduction=True):
    ly_count = 1
    x = stem(input, 'ly{}_'.format(str(ly_count)))
    ly_count += 1
    for i in range(4):
        x = inception_a(x, 'ly{}_'.format(str(ly_count)))
        ly_count += 1
    if enable_reduction:
        x = reduction_a(x, 'ly{}_'.format(str(ly_count)))
        ly_count += 1

    for i in range(7):
        x = inception_b(x, 'ly{}_'.format(str(ly_count)))
        ly_count += 1
    if enable_reduction:
        x = reduction_b(x, 'ly{}_'.format(str(ly_count)))
        ly_count += 1
    for i in range(3):
        x = inception_c(x, 'ly{}_'.format(str(ly_count)))
        ly_count += 1

    x = GlobalAveragePooling2D(data_format=DEFAULT_CHANNEL)(x)
    x = Dropout(0.8)(x)
    x = Dense(NB_OUTPUTS, activation='sigmoid', name='predictions')(x)

    return x

