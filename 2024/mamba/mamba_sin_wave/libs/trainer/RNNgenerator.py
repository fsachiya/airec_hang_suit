import numpy as np
import torch
import sys

# sys.path.append("/home/shigeki/Documents/sachiya/work/eipl/")

from eipl.utils import print_warn, print_info
from eipl.utils import tensor2numpy

class RNNgenerator:
    """
    Helper calass to generate rnn's prediction
    """
    def __init__(self, model, gpu=-1):
        if gpu >= 0:
            model = model.to('cuda:' + str(gpu))
        self.model = model
        
    def sequence_prediction(self, array, init_state=None, input_param=1.0):
        """
        Generates along with given array, and returns prediction, hidden states, and losses.
        Note that returned h includes initial states

        Example:
            array = (10, 2) with rnn which has ((1,5), (1,6)) hidden state
                --> returns y = (9, 2) and h = [(10,5), (10,6)]

        """

        h = None
        y_hist, h_hist, loss_hist = [], [], []
        for t in range(array.shape[0]):
            if t==0:
                x = array[t]
            else:
                x = input_param*array[t] + (1.0-input_param)*y_hist[-1]

            ### forward prop
            y, h = self.model.forward(x, h)
                        
            y_hist.append(y); h_hist.append(h)
            
        _y_hist, _h_hist = [], []
        for i in range(len(h)):
            _h_hist.append( np.vstack([ tensor2numpy(h[i]) for h in h_hist ]) )

        for _y in y_hist:
            _y_hist.append( tensor2numpy(_y) )
            
        return np.vstack(_y_hist), _h_hist

