zc2421, Zilin Chen
Date: 12/21/2019

Project Title: Speech Recognizer for KWS

Project summary:

   This project contains a lightweight implementation of single keyword speech recognition in Python that can be ran in local device with or without GPU. A convolutional recurrent neural network model (CRNN) is built from scratch using the keras API that runs with Tensorflow backend. The dataset is from Google Speech Command dataset https://ai.googleblog.com/2017/08/launching-speech-commands-dataset.html, which contains in the GSCdata directory. The implementation of CRNN is short and simple using the keras API. The model developed can achieve around 93% accuracy on the test dataset.


List of tools:

I used Python and run everything in local virtualenv. All the packages and libraries required to run my code are saved in requirements.txt. To install all requirements, simply run:
	pip install -r requirements.txt

List of directories and executables that may be used to test the code.

   To train new models run:
	python3 encoder.py

   To decode the model and see test results run:
	python3 decoder.py

The name(s) of one or two main scripts that run everything. Also, provide the sample output and where to look for it, if we run your example.
   
   -- encoder.py: train and save the models in current directory and plot accuracy/loss 		  graph. Graphs only displayed if uncomment #plt.show()
   -- decoder.py: decode models and return test results on train set, validation set, and 		  test sets. The two models used are already saved locally. If no need to 		  train another model, just run
			python3 decoder.py

	          Expected Output Format:
			Evaluation scores: 
			Metrics: ['loss', 'sparse_categorical_accuracy'] 
			Train: [0.11994876400831173, 0.9639033018867924] 
			Validation: [0.2253242879865631, 0.9412298387096775] 
			Test: [0.23988403107473558, 0.9348477964669886]



 