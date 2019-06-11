import numpy as np


class NeuralNetwork(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.sizes    = sizes
        self.biases   = [np.ones(n) for n in sizes[1:]]
        self.weights  = [np.ones((n,m)) for n,m in zip(sizes[:-1], sizes[1:])]

        for size in sizes:
            if size <= 0:
                print 'Any layer size should be > 0'
                assert(0)

    def generate(self, input):
        if len(input) != self.sizes[0]:
            print 'Input size (%d) should match first layer size (%d)' % (len(input), self.sizes[0])
            assert(0)

        m = []
        for l in xrange(self.n_layers-1):
            if l == 0:
                m = np.dot(np.array(input, dtype=float), self.weights[l]) - self.biases[l]
            else:
                m = np.dot(m, self.weights[l]) - self.biases[l]
        return m

if __name__=='__main__':
    nn = NeuralNetwork([3, 4, 2])
    output = nn.generate([0, 1, 2])