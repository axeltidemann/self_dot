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
        self.initial_state_inv = mdp.numx.zeros((1, self.input_dim))
        self.states_inv = mdp.numx.zeros((1, self.input_dim))
        self._inv_initialized = True
        
    def _inverse(self, y):
    
        if not self._inv_is_initialized:
            self.inv_initialize()

        if self.reset_states:
            self.initial_state_inv = mdp.numx.zeros((1, self.input_dim))
        else:
            self.initial_state_inv = mdp.numx.atleast_2d(self.states_inv[-1, :])

        steps = y.shape[0]
        
        states = mdp.numx.concatenate((self.initial_state_inv, mdp.numx.zeros((steps, self.input_dim))))

        nonlinear_function_pointer = self.nonlin_func

        for n in range(steps):
            states[n + 1, :] = nonlinear_function_pointer(mdp.numx.dot(self.w_inv, states[n, :]) + mdp.numx.dot(self.w_in_inv, y[n, :]) + self.w_bias_inv)
            self._post_update_hook(states, y, n)

        self.states_inv = states[1:]
        return self.states_inv

class InvertibleLinearRegressionNode(mdp.nodes.LinearRegressionNode):
    """Linear regression node that is invertible."""

    def is_invertible(self):
        return True

    def _inverse(self, y):
        return mdp.utils.mult(y, np.linalg.pinv(self.beta))


if __name__ == '__main__':
    readout = InvertibleLinearRegressionNode(with_bias=False)
    reservoir = InvertibleLeakyReservoirNode(output_dim=100, leak_rate=.8, bias_scaling=.2, reset_states=True)
    flow = mdp.hinet.FlowNode(reservoir+readout)
    x = np.random.rand(100,5)
    y = np.random.rand(100,3)
    flow.train(x,y)
    print Oger.utils.rmse(y, flow(x)), Oger.utils.rmse(x, flow.inverse(y))
