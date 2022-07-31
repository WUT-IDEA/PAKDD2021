def LSTM_fcn_channel_model():
    #     ip = Input(shape=(MAX_NB_VARIABLES, MAX_TIMESTEPS))
    ip = Input(shape=(sw_width, features))

    x = LSTM(100, activation='relu')(ip)
    #     x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    y = Conv1D(64, 8, padding='same', kernel_initializer='he_uniform')(ip)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = channel_attention(y)
    #     y = spatial_attention(y)

    y = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = channel_attention(y)
    #     y = spatial_attention(y)

    y = Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(pred_length, activation='relu')(x)

    model = Model(ip, out)
    model.summary()

    return model