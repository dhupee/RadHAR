
"""
directional LSTMS on VOXELS

Usage:

- extract_path is the where the extracted data samples are available.
- checkpoint_model_path is the path where to checkpoint the trained models during the training process


EXAMPLE: SPECIFICATION

extract_path = '/root/content/Data_data/Train_Data_voxels_'
checkpoint_model_path="/root/checkpoint"
"""

extract_path = '/root/content/Data_data/Train_Data_voxels_'
checkpoint_model_path="/root/checkpoint/LSTM"


import glob
import os
import numpy as np
# random seed.
rand_seed = 1
from numpy.random import seed
seed(rand_seed)
from tensorflow.compat.v1 import set_random_seed
#from tensorflow.keras.utils import set_random_seed
set_random_seed(rand_seed)

# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Activation
from tensorflow.keras.layers import Permute, Reshape
from tensorflow.keras import backend as K

from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import BatchNormalization
# from keras.layers.convolutional import *
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Bidirectional,TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model


sub_dirs=['boxing','jack','jump','squats','walk']

def one_hot_encoding(y_data, sub_dirs, categories=5):
    Mapping=dict()

    count=0
    for i in sub_dirs:
        Mapping[i]=count
        count=count+1

    y_features2=[]
    for i in range(len(y_data)):
        Type=y_data[i]
        lab=Mapping[Type]
        y_features2.append(lab)

    y_features=np.array(y_features2)
    y_features=y_features.reshape(y_features.shape[0],1)
    from tensorflow.keras.utils import to_categorical
    y_features = to_categorical(y_features)

    return y_features


def full_3D_model(summary=False):
    print('building the model ... ')
    model = Sequential()

    model.add(Bidirectional(LSTM(64, return_sequences=False, stateful=False,input_shape=(60, 10*1024) )))
    model.add(Dropout(.5,name='dropout_1'))
    model.add(Dense(128, activation='relu', name='DENSE_1'))
    model.add(Dropout(.5,name='dropout_2'))
    model.add(Dense(5, activation='softmax', name = 'output'))

    return model



frame_tog = [60]


#loading the train data
Data_path = extract_path+'boxing'

data = np.load(Data_path+'.npz')
train_data = data['arr_0']
train_data = np.array(train_data,dtype=np.dtype(np.int32))
train_label = data['arr_1']

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'jack'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)


del data

#print(train_data.shape,train_label.shape)


Data_path = extract_path+'jump'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'squats'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data
#print(train_data.shape,train_label.shape)

Data_path = extract_path+'walk'
data = np.load(Data_path+'.npz')
train_data = np.concatenate((train_data, data['arr_0']), axis=0)
train_label = np.concatenate((train_label, data['arr_1']), axis=0)

del data

train_label = one_hot_encoding(train_label, sub_dirs, categories=5)

train_data = train_data.reshape(train_data.shape[0],train_data.shape[1], train_data.shape[2]*train_data.shape[3]*train_data.shape[4])

print('Training Data Shape is:')
print(train_data.shape,train_label.shape)



X_train, X_val, y_train, y_val  = train_test_split(train_data, train_label, test_size=0.20, random_state=1)
del train_data,train_label


model = full_3D_model()


print("Model building is completed")


adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None,
                       decay=0.0, amsgrad=False)

loss_func = categorical_crossentropy

model.compile(loss=loss_func,
                   optimizer=adam,
                  metrics=['accuracy'])

checkpoint = ModelCheckpoint(checkpoint_model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

callbacks_list = [checkpoint]


# Training the model
learning_hist = model.fit(X_train, y_train,
                             batch_size=20,
                             epochs=30,
                             verbose=1,
                             shuffle=True,
                           validation_data=(X_val,y_val),
                           callbacks=callbacks_list
                          )

