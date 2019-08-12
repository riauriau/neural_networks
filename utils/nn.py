import numpy as np
import copy

class NeuralNetwork(object):
    def __init__(self, sizes):
        self.n_layers = len(sizes)
        self.sizes    = sizes
        self.biases   = [np.random.randn(n) for n in sizes[1:]]
        self.weights  = [np.random.randn(n,m) for n,m in zip(sizes[:-1], sizes[1:])]

        for size in sizes:
            if size <= 0:
                print 'Any layer size should be > 0'
                assert(0)

    def _sigmoid(self, x):
        return 1/(1+np.exp(-x))

    def _sigmoid_der(self, x):
        sigmoid = self._sigmoid(x)
        return sigmoid*(1. - sigmoid)

    def _feedfw(self, a):
        for b, w in zip(self.biases, self.weights):
            a = self._sigmoid(np.dot(a, w) + b)
        return a

    def _backprop(self):
        self

    def _train_batch(self, batch):
        pass

    def _evaluate(self, test_data):
        test_result = [(np.argmax(self._feedfw(test)), y) for (x, y) in test_data]

        ok_num = sum([x == y for (x, y) in test_result])
        test_num = len(test_data[0])
        results = {'test_num'      : test_num,
                   'ok_num'        : ok_num,
                   'nok_num'       : test_num - ok_num,
                   'ok_perc'       : 100. * ok_num / test_num,
                   'nok_perc'      : 100. * (test_num - ok_num) / test_num
                   }
        return results

    def train_network(self, train_data, iter, batch_size, learn_rate, test_data=None):
        train_data_ = copy.deepcopy(train_data) # keep original untouched
        train_data_size = len(train_data_)
        batch_size = min(batch_size, train_data_size)

        for j in iter:
            train_data_ = np.random.shuffle(train_data_)
            batch_data  = [train_data_[k:k+batch_size] for k in xrange(0, train_data_size)]

            for batch in batch_data:
                self._train_batch(batch, learn_rate)

            if test_data:
                test_results = self._evaluate(test_data)
                print "Epoch %d: Evaluation OK %d/%d (%.1f%%) NOK %d/%d (%.1f%%)" % (j, test_results['ok_num'], test_results['test_num'], test_results['ok_perc'],
                                                                                     test_results['nok_num'], test_results['test_num'], test_results['nok_perc'])
            else:
                print "Epoch %d" % (j)


if __name__=='__main__':
    nn = NeuralNetwork([3, 4, 2])
    output = nn.generate([0, 1, 2])