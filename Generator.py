#Zilin Chen
#zc2421

import numpy as np
import random
from keras.utils import Sequence

#implement data generator that generates a sequence of data
#class extend the Sequence Object
#Every Sequence must implement the __getitem__ and the __len__ methods. If you want to modify your dataset between epochs you may implement on_epoch_end
#methods templates copied from original source codes https://github.com/keras-team/keras/blob/master/keras/utils/data_utils.py#L305

class GenDataSequence(Sequence):

    def __init__(self, fids, flabels, batch=64, shuffle=True):
        self.batch = batch
        self.flabels = flabels
        self.fids = fids
        self.shuffle = shuffle
        self.on_epoch_end()


    # get current batch size
    def getBatch(self):
        return self.batch

    # change the batch size of the data sequence
    def setBatch(self, batch_size):
        self.batch = batch_size

    #get the class id of a audio file
    def getClassId(self, index):
        return self.flabels[self.fids[index]]


    #abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.
        # Returns
            The number of batches in the Sequence.
        """
        return len(self.fids) // self.batch
        

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        self.idx_list = np.arange(len(self.fids))
        if self.shuffle:
            np.random.shuffle(self.idx_list)

    #abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.
        # Arguments
            index: position of the batch in the Sequence.
        # Returns
            A batch
        """
        #get list of file ids
        l = index * self.batch
        r = (index+1)*self.batch
        f_list = self.idx_list[l:r]
        fids = [self.fids[i] for i in f_list]

        # Generate data
        dim = 16000
        audios = np.empty((self.batch, dim))
        cid = np.empty((self.batch))
        
        # Generate audio data
        for i in range(len(fids)):
            temp = np.load(fids[i])
            length = temp.shape[0]
            #default dimension is 16000 for google speech command dataset
            if length == dim:
                audios[i] = temp
            elif length > dim:
                #trunk length
                temp_idx = random.randint(0,length-dim)
                audios[i] = temp[temp_idx:temp_idx+dim]
            else: 
                temp_idx = random.randint(0,dim-length)
                audios[i,temp_idx:temp_idx+length] = temp
            
            # get class id
            cid[i] = int(self.flabels[fids[i]])

        return audios, cid




