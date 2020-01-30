from keras.models import Model
from keras.layers import Input, Permute, Reshape, Lambda, Dot, Softmax
from keras.layers import BatchNormalization, Conv2D, Reshape,Dense, LSTM, CuDNNLSTM, Bidirectional
from keras import backend as K
from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D
import random

######source code############
def computeMelspectrogram(inputs, samplingrate=16000, inputLength = 16000):
    
    x = Reshape((1, -1)) (inputs)
    x = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, inputLength),
                             padding='same', sr=samplingrate, n_mels=80,
                             fmin=40.0, fmax=samplingrate/2, power_melgram=1.0,
                             return_decibel_melgram=True, trainable_fb=False,
                             trainable_kernel=False,
                             name='mel_scale_spectrogram')(x)
    x = Normalization2D(int_axis=0)(x)
    x = Permute((2,1,3)) (x)
    return x

##################################

#covolutional recurrent model
def CRNNModel(CuDNN = False):
    # generate input layer
    # instantiate a Keras tensor object
    inputs = Input((16000,))
    
    #generate mel scale spectrogramx
    mel = computeMelspectrogram(inputs)

    #zc2421
    #build 2 convolutional layers
    x = Conv2D(filters=10, kernel_size=(5,1) , padding='same', activation='relu', use_bias=True)(mel)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Conv2D(filters=1, kernel_size=(5,1) , padding='same', activation='relu', use_bias=True) (x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Reshape((125, 80)) (x)
    #specify training functions
    if not CuDNN: 
        func = LSTM
    else:
        func = CuDNNLSTM
    #build 2 lstm layers
    x = Bidirectional(func(64, return_sequences = True))(x) 
    x = Bidirectional(func(64)) (x)
    #construct 3 fully connected layers
    x = Dense(64, activation = 'relu')(x)
    x = Dense(32, activation = 'relu')(x)
    output = Dense(36, activation = 'softmax')(x)
    model = Model(inputs=inputs, outputs=output)    
    return model

#attention model
def AttCRNNModel(CuDNN = False):  
    inputs = Input((16000,))
    
    #generate mel scale spectrogramx
    mel = computeMelspectrogram(inputs)

    #build 2 convolutional layers
    x = Conv2D(filters=10, kernel_size=(5,1) , padding='same', activation='relu', use_bias=True)(mel)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001) (x)
    x = Conv2D(filters=1, kernel_size=(5,1) , padding='same', activation='relu', use_bias=True) (x)
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001) (x)
    x = Reshape((125, 80)) (x)
    if not CuDNN: 
        func = LSTM
    else:
        func = CuDNNLSTM
    #build 2 lstm layers
    x = Bidirectional(func(64, return_sequences = True)) (x) 
    x = Bidirectional(func(64, return_sequences = True)) (x) 

    #calculate attention layer
    #v_idx = random.randint(0,128)
    v_idx = 80
    att_v = Lambda(lambda x: x[:,v_idx]) (x) 
    att_v = Dense(128) (att_v)
    att_scores = Dot(axes=[1,2])([att_v, x]) 
    att_scores = Softmax(name='att_weight')(att_scores) 
    att_weight = Dot(axes=[1,1])([att_scores, x]) 

    #build dense layer
    x = Dense(64, activation = 'relu')(att_weight)
    x = Dense(32)(x)
    output = Dense(36, activation = 'softmax', name='att_output')(x)
    model = Model(inputs=inputs, outputs=output) 
    return model