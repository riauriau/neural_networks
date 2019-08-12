from utils.nn import NeuralNetwork
from utils.mnist import Mnist

import matplotlib.pyplot as plt

def train_network(nn, mnist):

    LEARN_RATE = 0.1
    BATCH_SIZE = 100
    ITERATIONS = 100
    VALIDATION_SET_SIZE = 10000
    training_set_size = len(mnist.training_set_images) - VALIDATION_SET_SIZE

    train_data = (mnist.training_set_images[0:training_set_size], mnist.training_set_images[0:training_set_labels])
    valid_data = (mnist.training_set_images[training_set_size:], mnist.training_set_images[training_set_labels:])
    test_data  = (mnist.test_set_images, test_set_labels)

    nn.train_network(train_data, ITERATOINS, BATCH_SIZE, LEARN_RATE, test_data)

if __name__=="__main__":
    training_set = [r'C:\git\neural_networks\mnist\train-images.idx3-ubyte.data', r'C:\git\neural_networks\mnist\train-labels.idx1-ubyte.data']
    test_set = [r'C:\git\neural_networks\mnist\t10k-images.idx3-ubyte.data', r'C:\git\neural_networks\mnist\t10k-labels.idx1-ubyte.data']
    mnist = Mnist(training_set, test_set)

    [training_images, training_labels] = mnist.get_training_set()
    INPUT_LAYER_SIZE = training_images.shape[1] * training_images.shape[2]
    OUTPUT_LAYER_SIZE = 9
    HIDDEN_LAYERS = [100, 200]
    layers = []
    layers.append(INPUT_LAYER_SIZE)
    [layers.append(element) for element in HIDDEN_LAYERS]
    layers.append(OUTPUT_LAYER_SIZE)
    nn = NeuralNetwork(layers)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    ax.set_title('First image')
    plt.imshow(training_images[0], cmap='gray_r')
    ax.set_aspect('equal')
    plt.show()

    train_network(nn, mnist)