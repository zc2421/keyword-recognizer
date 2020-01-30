#Zilin Chen
#zc2421

import pandas
import os

#read: and store data into variable
#output: data dictionary
def processData():
	#get file list
	gsc = 'GSCdata'
	valList = pandas.read_csv(gsc+'/train/validation_list.txt', sep=" ", header=None)[0].tolist()
	valList  = [os.path.join(gsc+'/train/', f + '.npy') for f in valList if f.endswith('.wav')]
	testList = pandas.read_csv(gsc+'/train/testing_list.txt', sep=" ", header=None)[0].tolist()
	testList = [os.path.join(gsc+'/train/', f + '.npy') for f in testList if f.endswith('.wav')]
	trainList = []
	for r, _, files in os.walk(gsc+'/train/'):
	    trainList += [r+'/'+ f for f in files if f.endswith('.wav.npy')]
	trainList = list( set(trainList)-set(valList)-set(testList) )
	testRList = []
	for r, _, files in os.walk(gsc+'/test/'):
	    testRList += [r+'/'+ f for f in files if f.endswith('.wav.npy')]


	#getting labels for file
	trainLable, valLable, testLabel, testRLabel = [], [], [], []
	for f in trainList:
		trainLable.append(getFileCat(f))
	for f in valList:
		valLable.append(getFileCat(f))
	for f in testList:
		testLabel.append(getFileCat(f))
	for f in testRList:
		testRLabel.append(getFileCat(f))


	#hashmap from file to its category
	trainDict = dict(zip(trainList, trainLable))
	valDict = dict(zip(valList, valLable))
	testDict = dict(zip(testList, testLabel))
	testRDict = dict(zip(testRList, testRLabel))

	#format output data
	train = {'files' : trainList, 'labels' : trainDict}
	val = {'files' : valList, 'labels' : valDict}
	test = {'files' : testList, 'labels' : testDict}
	testR = {'files' : testRList, 'labels' : testRDict}
	gData = {'train' : train, 'val' : val, 'test' : test, 'testR' : testR}    

	return gData

#get the classification category of the file
#input: file
#output: category
def getFileCat(f):
	class2Idx = {'unknown' : 0, 'silence' : 0, '_unknown_' : 0, '_silence_' : 0, '_background_noise_' : 0, 'nine' : 1, 'yes' : 2, 
			'no' : 3, 'up' : 4, 'down' : 5, 'left' : 6, 'right' : 7, 'on' : 8, 'off' : 9, 'stop' : 10, 'go' : 11,
			'zero' : 12, 'one' : 13, 'two' : 14, 'three' : 15, 'four' : 16, 'five' : 17, 'six' : 18, 
			'seven' : 19,  'eight' : 20, 'backward':21, 'bed':22, 'bird':23, 'cat':24, 'dog':25,
			'follow':26, 'forward':27, 'happy':28, 'house':29, 'learn':30, 'marvin':31, 'sheila':32, 'tree':33,
			'visual':34, 'wow':35}

	fn = os.path.dirname(f)
	cat = os.path.basename(fn)
	# categorize all slience, noise, unknown to 0
	return class2Idx.get(cat,0)


	