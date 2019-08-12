import os, sys
import idx2numpy

class Mnist(object):
    def __init__(self, training_set, test_set):
        self.training_set_images = training_set[0]
        self.training_set_labels = training_set[1]
        self.test_set_images = test_set[0]
        self.test_set_labels = test_set[1]

    def get_training_set(self):
        return (idx2numpy.convert_from_file(self.training_set_images), idx2numpy.convert_from_file(self.training_set_labels))

    def get_test_set(self):
        return (idx2numpy.convert_from_file(self.test_set_images), idx2numpy.convert_from_file(self.test_set_labels))


if __name__=="__main__":
    training_set = [r'C:\git\neural_networks\mnist\train-images.idx3-ubyte.data', r'C:\git\neural_networks\mnist\train-labels.idx1-ubyte.data']
    test_set = [r'C:\git\neural_networks\mnist\t10k-images.idx3-ubyte.data', r'C:\git\neural_networks\mnist\t10k-labels.idx1-ubyte.data']

    mnist = Mnist(training_set, test_set)
    [test_images, test_labels] = mnist.get_test_set()
    [training_images, training_labels] = mnist.get_training_set()
    pass