#Zilin Chen
#zc2421

import Generator
import models
import prepareData
import utils
import sklearn
from sklearn.metrics import confusion_matrix


#device
CuDNN = False


#build data sequence generator
gscData = prepareData.processData()
genTrain = Generator.GenDataSequence(gscData['train']['files'], gscData['train']['labels'], shuffle = True)
genVal = Generator.GenDataSequence(gscData['val']['files'], gscData['val']['labels'], shuffle = True)
test_batch = len(gscData['test']['files'])
genTest = Generator.GenDataSequence(gscData['test']['files'], gscData['test']['labels'], batch = test_batch, shuffle = False)
testR_batch = len(gscData['testR']['files'])
genTestR = Generator.GenDataSequence(gscData['testR']['files'], gscData['testR']['labels'], batch = testR_batch, shuffle = False)


#get all test data of 35 keywords
x_test, y_test = genTest.__getitem__(0)
# #get all test data of 10 keywords
# x_test, y_test = genTestR.__getitem__(0)

#retrain rnn model
rnn_model = models.CRNNModel(CuDNN=CuDNN)
rnn_model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
#load rnn model with best weights
rnn_model.load_weights('rnn_model.h5')
#evaluate rnn model
valScore = rnn_model.evaluate_generator(genVal, use_multiprocessing=True, workers=4, verbose=1)
trainScore = rnn_model.evaluate_generator(genTrain, use_multiprocessing=True, workers=4, verbose=1)
testScore = rnn_model.evaluate(x_test, y_test, verbose=1)
print('Evaluation scores: \nMetrics: {} \nTrain: {} \nValidation: {} \nTest: {}'.format(rnn_model.metrics_names, trainScore, valScore, testScore) )

#retrain att-rnn model
att_rnn_model = models.AttCRNNModel(CuDNN=CuDNN)
att_rnn_model.compile(optimizer='adam', loss=['sparse_categorical_crossentropy'], metrics=['sparse_categorical_accuracy'])
#load model with best weights
att_rnn_model.load_weights('rnn_att_model.h5')

#evaluate model
valScore = att_rnn_model.evaluate_generator(genVal, use_multiprocessing=True, workers=4,verbose=1)
trainScore = att_rnn_model.evaluate_generator(genTrain, use_multiprocessing=True, workers=4,verbose=1)
testScore = att_rnn_model.evaluate(x_test, y_test, verbose=1)
print('Attention Evaluation scores: \nMetrics: {} \nTrain: {} \nValidation: {} \nTest: {}'.format(att_rnn_model.metrics_names, trainScore, valScore, testScore) )


#optinal: plot attention graph
# utils.plotAttention(att_rnn_model, test, 0):



