import theano
import theano.tensor as T
import numpy as np


class HighwayNetwork:

	""" Implementation of Highway network from paper
        Highway Networks http://arxiv.org/abs/1505.00387
    """

	def __init__(self, x=None, labels=None, theano_input=None, num_vis=100, num_hid=50, numpy_rng=None, theano_rng=None, lrate=0.001, , momentum=0.9, tied_weights=True, cost_func='ce'):
		'''
			x - input data in simple numpy ndarray format
            y - labels
			theano_input - input data in theano format (tensor.matrix, tensor.vector, etc..)
			num_vis - number of visible units
			num_hid - number of hidden units
			numpy_rng - generator for random numbers
			lrate - learning rate for training
			momentum - for training to help out from local minima (TODO: Nesterov momentum)
			cost_func - cost function(ce, lse)
			(bouth can be None)
		'''
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
        self.y = T.matrix('x')
        self.labels = labels
		w_init = 4 * np.sqrt(6. / (num_hid + num_vis))
		par = ParametersInit(self.numpy_rng, -w_init, w_init)
	    self.Wt = par.get_weights((num_vis, num_hid),'Wt')
	    self.Wc = par.get_weights((num_vis, num_hid),'Wc')
		# Bias init as zero
		self.bh = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh')
		self.bh2 = theano.shared(np.asarray(np.zeros(num_hid), dtype=theano.config.floatX), name='bh2')
		self.bv = theano.shared(np.asarray(np.zeros(num_vis), dtype=theano.config.floatX), name='bv')
		self.params = [self.Wt, self.Wc, self.bv, self.bh, self.bh2]
		self.lrate = lrate
		self.corruption_level = corruption_level
		self.num_vis = num_vis
		self.num_hid = num_hid
		self.momentum = momentum
		self.cost_func = cost_func
		self._set_encoder_func(encoder_func, decoder_func)
		self.params = []

    def forward(self):
        """ Basic forward step """:
            carry = T.tahn(T.dot(self.x, self.Wc) +self.bh2)
            return T.nnet.sigmoid(T.dot(self.x, self.Wt) + self.bh) * carry + x * (1 - carry)

    def cost_value(self):
        output = self.forward()
        cost = -T.sum(output * T.log(self.y) + (1 - output) * T.log(1 - self.y), axis=1)
        grads = T.grad(cost, self.params)
        return cost, [(oldvalue, oldvalue - self.lrate * newvalue) for (oldvalue, newvalue) in zip(self.params, grads)]

    def training(self, iters=200):
        cost, updates = self.cost_value()
        func = theano.function([], cost,updates=updates,givens={self.x: self.inp, self.y: self.labels})
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
