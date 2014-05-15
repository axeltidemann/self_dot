import Oger
import mdp
import numpy as np

class InvertibleLeakyReservoirNode(Oger.nodes.LeakyReservoirNode):
    """Leaky reservoir node that is invertible."""

    def __init__(self, *args, **kwargs):
        super(InvertibleLeakyReservoirNode, self).__init__(*args, **kwargs)
        self._inv_is_initialized = False
        
    def is_invertible(self):
        return True

    def inv_initialize(self):
        self.w_in_inv = np.linalg.pinv(self.w_in)
        self.w_inv = np.linalg.pinv(self.w)
        self.w_bias_inv = np.linalg.pinv(self.w_bias)
        self.initial_x = mdp.numx.zeros((1, self.input_dim))
        self.x = mdp.numx.zeros((1, self.input_dim))
        self._inv_initialized = True
        
    def _inverse(self, y):
    
        if not self._inv_is_initialized:
            self.inv_initialize()

        if self.reset_states:
            self.initial_x = mdp.numx.zeros((1, self.input_dim))
        else:
            self.initial_x = mdp.numx.atleast_2d(self.x[-1])

        steps = y.shape[0]
        
        x = mdp.numx.concatenate((self.initial_x, mdp.numx.zeros((steps, self.input_dim))))

        nonlinear_function_pointer = self.nonlin_func

        import ipdb
        for n in range(steps):
            #ipdb.set_trace()
            #x[n + 1] = nonlinear_function_pointer(mdp.numx.dot(self.w_inv, y[n]) +  self.w_bias_inv)mdp.numx.dot(self.w_in_inv, x[n])
            x[n + 1] = nonlinear_function_pointer(mdp.numx.dot( mdp.numx.dot(self.w_inv, y[n] ) + self.w_bias, self.w_in ))
            self._post_update_hook(x, y, n)

        self.x = x[1:]
        return self.x

class InvertibleLinearRegressionNode(mdp.nodes.LinearRegressionNode):
    """Linear regression node that is invertible."""

    def is_invertible(self):
        return True

    def _inverse(self, y):
        return mdp.utils.mult(y, np.linalg.pinv(self.beta))


if __name__ == '__main__':
    readout = InvertibleLinearRegressionNode(with_bias=False)
    reservoir = InvertibleLeakyReservoirNode(output_dim=100, leak_rate=.8, bias_scaling=.2, reset_states=True, nonlin_func=lambda x: x)
    flow = mdp.hinet.FlowNode(reservoir+readout)
    x = np.random.rand(100,5)
    y = np.random.rand(100,3)
    flow.train(x,y)
    print Oger.utils.rmse(y, flow(x)), Oger.utils.rmse(x, flow.inverse(y))
