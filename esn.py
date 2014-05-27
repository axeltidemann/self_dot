from warnings import warn

import Oger
import mdp
import numpy as np

class ACDCESN:
    '''An Echo State Network that can be used in both directions. It
    achieves this by having two networks. Upon execution the correct
    network is selected depending on the input dimensions. If input/output signals
    are of the same dimension, you must use inverse() to get flow in the
    opposite direction.'''

    def __init__(self, *args, **kwargs):
        self.ac = self._flow(*args, **kwargs)
        self.dc = self._flow(*args, **kwargs)

    def __call__(self, x):
        try:
            return self.ac(x)
        except:
            return self.dc(x)

    def _flow(self, hidden_nodes, leak_rate, bias_scaling, reset_states, use_pinv):
        reservoir = Oger.nodes.LeakyReservoirNode(output_dim=hidden_nodes, 
                                                  leak_rate=leak_rate, 
                                                  bias_scaling=bias_scaling, 
                                                  reset_states=reset_states)
        readout = mdp.nodes.LinearRegressionNode(use_pinv=use_pinv)
        return mdp.hinet.FlowNode(reservoir + readout)

    def train(self, x, y):
        self.ac.train(x,y)
        self.dc.train(y,x)

        if x.shape[1] == y.shape[1]:
            warn('Input and output signals are of the same dimension, ' +
                 'you must explicitly use inverse() to get flow in the ' +
                 'opposite direction.')

    def inverse(self, x):
        return self.dc(x)

if __name__ == '__main__':
    myACDC = ACDCESN(hidden_nodes=100, leak_rate=.8, bias_scaling=.2, reset_states=True, use_pinv=True)
    x = np.random.rand(100,5)
    y = np.random.rand(100,7)
    myACDC.train(x,y)
    print Oger.utils.rmse(y, myACDC(x)), Oger.utils.rmse(x, myACDC(y))
