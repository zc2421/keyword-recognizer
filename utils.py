#Zilin Chen
#zc2421

import math
import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt

#update the learning rate for each epoch during training
#input: epoch number
#output: learning rate
def update(epoch):
    init = 0.001
    drop = 0.4
    epochs_drop = 10.0
    new_rate = init * math.pow(drop, math.floor((1+epoch)/epochs_drop))  
    print('Learning rate {}'.format(new_rate))
    return new_rate


# plot attention weight
# input: att_model, audios for prediction, index of the audio to plot
# output: None
def plotAttention(att_model, audios, idx):
	inp = att_model.input
	out = [att_model.get_layer('att_weight').output,
			   att_model.get_layer('mel_scale_spectrogram').output]

	temp_Model = Model(inputs = inp, outpus = out)
	att_weight, spect = temp_Model.predict(audios)

	#plot attention 
	plt.figure(figsize=(10,3))
	plt.title("Attention")
	plt.ylabel('Attention Weight log')
	plt.xlabel('Index')
	plt.plot(np.log(att_weight[idx]))
	# plt.show()

	#plot raw wave
	plt.figure(figsize=(10,3))
	plt.title('Raw waveform')
	plt.ylabel('Amplitude')
	plt.xlabel('Sampling Rate Index')
	plt.plot(audios[idx])
	# plt.show()

	#plot mel-scale spectrogram
	plt.figure(figsize=(10,3))
	plt.title('Mel Spectrogram')
	plt.ylabel('Frequency')
	plt.xlabel('Time')
	plt.pcolormesh(spect[idx,:,:,0])
	# plt.show()



