import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np


class HighwayNetwork:
    """ Implementation of Highway network from the paper
    Highway Networks http://arxiv.org/abs/1505.00387
    """

    def __init__(self, x=None, labels=None, theano_input=None, num_vis=100, num_hid=50, num_out=1, numpy_rng=None, theano_rng=None, lrate=0.001, momentum=0.9, tied_weights=True, cost_func='ce'):
            self.training = x
            if theano_input == None:
                self.x = T.matrix('x')
            else:
                self.x = theano_input
            self.x = T.matrix('x')
            self.numpy_rng = numpy_rng
            if numpy_rng == None:
                self.numpy_rng = np.random.RandomState()
            self.theano_rng = theano_rng
            if theano_rng == None:
                self.theano_rng = RandomStreams()
            self.y = T.vector('y')
            self.labels = labels
            w_init = 4 * np.sqrt(6. / (num_hid + num_vis))
            par = ParametersInit(self.numpy_rng, -w_init, w_init)
            self.Wh = par.get_weights((num_vis, num_hid), 'Wh')
            self.Wt = par.get_weights((num_hid, num_hid), 'Wt')
            self.Wc = par.get_weights((num_hid, num_hid), 'Wc')
            self.Wo = par.get_weights((num_hid, num_out), 'Wo')
            # Bias init as zero
            self.bh = theano.shared(
                    np.asarray(np.random.random((num_hid,))),name='bh')
            self.bh2 = theano.shared(
                    np.asarray(np.random.random((num_hid,))),name='bh2')
            self.bh3 = theano.shared(
                    np.asarray(np.zeros(num_hid)), name='bv')
            self.bv = theano.shared(
                    np.asarray(np.zeros(num_vis)), name='bv')
            self.bo = theano.shared(
                    np.asarray(np.zeros(num_out)), name='bv')
            self.params = [self.Wt, self.Wc, self.Wo, self.bh, self.bh2, self.bo, self.bh3, self.Wh]
            self.lrate = lrate
            self.num_vis = num_vis
            self.num_hid = num_hid
            self.momentum = momentum

    def forward(self):
        hidden = T.nnet.sigmoid(T.dot(self.x, self.Wh) + self.bh3)
        carry = T.tanh(T.dot(hidden, self.Wc) + self.bh)
        gate = T.nnet.sigmoid(T.dot(hidden, self.Wt) + self.bh2)
        result = gate * carry + (1 - carry) * hidden
        return T.nnet.softmax(T.dot(result,self.Wo) + self.bo)

    def cost_value(self):
        output = self.forward()
        cost = T.nnet.binary_crossentropy(output, self.y).mean()
        grads = T.grad(T.mean(cost), self.params)
        return cost, [(oldvalue, oldvalue - self.lrate * newvalue) for (oldvalue, newvalue) in zip(self.params, grads)]

    def train(self, iters=200):
        cost, updates = self.cost_value()
        func = theano.function([], cost,updates=updates,givens={self.x: self.training, self.y: self.labels})
        for i in range(iters):
            cost_result = func()
            print("Cost {0}".format(cost_result))

    def predict(self, X):
        pass


class ParametersInit:
        def __init__(self, rng, low, high):
                """
                        rng - theano or numpy
                        low - low value for init
                        high - high value for init
                """
                self.rng = rng
                self.low = low
                self.high = high

        def get_weights(self, size, name, init_type='standard'):
                """
                        size in tuple format
                        name - current name for weights
                """
                if init_type == 'xavier':
                        return self._initW2(size, self.low, self.high, name)
                else:
                        return self._initW(size, name)


        def _initW(self, size, name):
                return theano.shared(value = \
                        np.asarray(
                                self.rng.uniform(low=self.low, high=self.high, size=size), dtype=theano.config.floatX
                                ))

        def _initW2(self, size, nin, nout, name):
                return theano.shared(value = \
                        np.asarray(
                                self.rng.uniform(low=-np.sqrt(6)/np.sqrt(nin + nout), high=np.sqrt(6)/np.sqrt(nin + nout), \
                                        size=size), dtype=theano.config.floatX
                                ), name=self.name)
