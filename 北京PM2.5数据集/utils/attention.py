from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, GlobalAveragePooling2D, GlobalMaxPooling2D, Reshape, Dense, multiply, Permute, Concatenate, Conv2D, Conv1D, Add, Activation, Lambda
from keras import backend as K
from keras.activations import sigmoid

'''
attetion block including spatial_attention block and channel_attention block
'''

def spatial_attention(input_feature):
    kernel_size = 7

    avg_pool = Lambda(lambda x: K.mean(x, axis=2, keepdims=True))(input_feature)
    spatial_feature = Conv1D(filters = 1,
                    kernel_size=kernel_size,
                    strides=1,
                    padding='same',
                    activation='sigmoid',
                    kernel_initializer='he_normal',
                    use_bias=False)(avg_pool)
    assert spatial_feature._keras_shape[-1] == 1

#     if K.image_data_format() == "channels_first":
#         cbam_feature = Permute((3, 1, 2))(cbam_feature)

    return multiply([input_feature, spatial_feature])

def channel_attention(input):

    filters = input._keras_shape[-1] # channel_axis = -1 for TF

    se = GlobalAveragePooling1D()(input)
    se = Reshape((1, filters))(se)
    se = Dense(filters // 16,  activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    se = multiply([input, se])
    return se