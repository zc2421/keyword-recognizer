#Zilin Chen
#zc2421

import prepareData
import Generator
import models
import utils
import tensorflow as tf
import librosa
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras_tqdm import TQDMNotebookCallback


idx2Class = {0: 'unknown_silence_noise', 1: 'nine', 2: 'yes', 3: 'no', 4: 'up', 5: 'down', 6: 'left', 7: 'right', 8: 'on', 9: 'off', 10: 'stop', 11: 'go', 
			12: 'zero', 13: 'one', 14: 'two', 15: 'three', 16: 'four', 17: 'five', 18: 'six', 19: 'seven', 20: 'eight', 21: 'backward', 22: 'bed', 23: 'bird', 
			24: 'cat', 25: 'dog', 26: 'follow', 27: 'forward', 28: 'happy', 29: 'house', 30: 'learn', 31: 'marvin', 32: 'sheila', 33: 'tree', 34: 'visual', 35: 'wow'}


# load data and its categories
gscData = prepareData.processData()
print('#train files:', len(gscData['train']['files']))
print('#test files:', len(gscData['test']['files']))
print('#validation files:', len(gscData['val']['files']))


genTrain = Generator.GenDataSequence(gscData['train']['files'], gscData['train']['labels'], shuffle = True)
genVal = Generator.GenDataSequence(gscData['val']['files'], gscData['val']['labels'], shuffle = True)

#plot a random word
audios, classes = genVal.__getitem__(3)
index = random.randint(1, len(classes))
plt.plot(audios[index])
word = idx2Class[classes[index]]
plt.title(word)
# plt.show()


# #build rnn model
loss_cat = 'sparse_categorical_crossentropy'
acc_met = 'sparse_categorical_accuracy'
val_acc_met = 'val_sparse_categorical_accuracy'
epochs = 25

rnn_model = models.CRNNModel(CuDNN = False)
# rnn_model.summary()
rnn_model.compile(optimizer='adam', loss=[loss_cat], metrics=[acc_met])
f_rnn = 'rnn_model.h5'
#train rnn_model
f_learningrate = LearningRateScheduler(utils.update, verbose=0)
f_earlystopping = EarlyStopping(monitor=val_acc_met, min_delta=0, patience=5, verbose=1, mode='auto', baseline=None, restore_best_weights=False)
f_checkpointer = ModelCheckpoint(f_rnn, monitor=val_acc_met, verbose=1, save_weights_only=True, save_best_only=True)
result1 = rnn_model.fit_generator(genTrain, validation_data = genVal, epochs = epochs, verbose=1, callbacks=[f_learningrate, f_earlystopping, f_checkpointer, TQDMNotebookCallback()])

#build train rnn att model
f_att_rnn = 'rnn_att_model.h5'
rnn_att_model = models.AttCRNNModel(CuDNN = False)
# rnn_att_model.summary()
rnn_att_model.compile(optimizer='adam', loss=[loss_cat], metrics=[acc_met])
f_checkpointer = ModelCheckpoint(f_att_rnn, monitor=val_acc_met, verbose=1, save_weights_only=True, save_best_only=True)
result2 = rnn_att_model.fit_generator(genTrain, validation_data = genVal, epochs = epochs, verbose=1, callbacks=[f_learningrate, f_earlystopping, f_checkpointer, TQDMNotebookCallback()])

#make accuracy plots for CRNN model
# print (result1.history.keys())
acc = result1.history['sparse_categorical_accuracy']
val_acc = result1.history['val_sparse_categorical_accuracy']
plt.plot(acc)
plt.plot(val_acc)
plt.title('CRNN Categorical accuracy')
plt.xlabel('#epoch')
plt.ylabel('accuracy')
plt.legend(['train', 'test'], loc=0)
# plt.show()

#make loss plots for CRNN model
loss = result1.history['loss']
val_loss = result1.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title('CRNN Model loss')
plt.ylabel('loss')
plt.xlabel('#epoch')
plt.legend(['train', 'test'], loc=0)
# plt.show()

#make accuracy plots for CRNN + Attention Model
# print (result2.history.keys())
acc = result2.history['sparse_categorical_accuracy']
val_acc = result2.history['val_sparse_categorical_accuracy']
plt.plot(acc)
plt.plot(val_acc)
plt.title('CRNN Attention Categorical accuracy')
plt.xlabel('#epoch')
plt.ylabel('%accuracy')
plt.legend(['train', 'test'], loc=0)
# plt.show()

#make loss plots for CRNN + Attention Model
loss = result2.history['loss']
val_loss = result2.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title('CRNN Attention Model loss')
plt.ylabel('loss')
plt.xlabel('#epoch')
plt.legend(['train', 'test'], loc=0)
# plt.show()


