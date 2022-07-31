from keras.models import Model
from keras.layers import Input, Dense, LSTM, multiply, concatenate, Activation, Masking, Reshape
from keras.layers import Conv1D, BatchNormalization, GlobalAveragePooling1D, Permute, Dropout

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from utils.attention import channel_attention,spatial_attention
from utils.data_split import split_sequence_parallel

# 不加这几句，则CONV 报错
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# tf.config.experimental.set_memory_growth(physical_devices[0], True)

# import sys


data = pd.read_csv("./data/data_2010.1.1-2014.12.31.csv", header=0,infer_datetime_format=True, engine='python')
data.drop('Unnamed: 0',axis=1, inplace=True)
data['datatime']=pd.to_datetime(data['datatime'])
data.set_index("datatime",inplace=True)

'''
归一化
'''
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
scaler2 = MinMaxScaler(feature_range=(0, 1))
scaler3 = MinMaxScaler(feature_range=(0, 1))
scaler4 = MinMaxScaler(feature_range=(0, 1))
scaler5 = MinMaxScaler(feature_range=(0, 1))
scaler6 = MinMaxScaler(feature_range=(0, 1))
scaler7 = MinMaxScaler(feature_range=(0, 1))

data_minmax = data.copy()
data_minmax['pm2.5']=scaler1.fit_transform(data_minmax['pm2.5'].values.reshape(-1,1))
data_minmax['DEWP']=scaler2.fit_transform(data_minmax['DEWP'].values.reshape(-1,1))
data_minmax['TEMP']=scaler3.fit_transform(data_minmax['TEMP'].values.reshape(-1,1))
data_minmax['PRES']=scaler4.fit_transform(data_minmax['PRES'].values.reshape(-1,1))
data_minmax['Iws']=scaler5.fit_transform(data_minmax['Iws'].values.reshape(-1,1))
data_minmax['Is']=scaler6.fit_transform(data_minmax['Is'].values.reshape(-1,1))
data_minmax['Ir']=scaler7.fit_transform(data_minmax['Ir'].values.reshape(-1,1))

'''
前四年训练加验证，最后一年测试
'''
cast = 35064
data_train = data_minmax[:cast]
data_test = data_minmax[cast:]
data_train = np.array(data_train)
data_test = np.array(data_test)


"""
滑动窗口及预测长度
"""
sw_width = 14
pred_length = 1
verbose_set = 2
X, y, features = split_sequence_parallel(data_train,sw_width,pred_length)
test_x,test_y,test_features = split_sequence_parallel(data_test,sw_width,pred_length)


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
    # y = spatial_attention(y)

    y = Conv1D(32, 5, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = channel_attention(y)
    # y = spatial_attention(y)

    y = Conv1D(64, 3, padding='same', kernel_initializer='he_uniform')(y)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    y = GlobalAveragePooling1D()(y)

    x = concatenate([x, y])

    out = Dense(pred_length, activation='relu')(x)

    model = Model(ip, out)
    model.summary()

    return model

LSTM_fcn_channel_model1 = LSTM_fcn_channel_model()

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

filepath = ".\weights\LSTM_FCN_spatial_model_weights_best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min', period=1)
callbacks_list = [checkpoint]
starttime=time.time()
LSTM_fcn_channel_model1.compile(loss=root_mean_squared_error, optimizer='adam')
history1 = LSTM_fcn_channel_model1.fit(X,y,validation_split=0.25,epochs=100, batch_size=32,callbacks=callbacks_list, verbose=2)
endtime=time.time()
dtime=endtime-starttime
print("程序运行时间为：%.8s s" % dtime)  #时间显示到微秒

loss = history1.history['loss']
val_loss = history1.history['val_loss']

epochs = np.arange(300) + 1
plt.figure(figsize=(20,18))
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title("Effect of model capacity on validation loss\n")
plt.xlabel('Epoch #')
plt.ylabel('Validation Loss')
plt.legend()
plt.show()