import numpy as np
from abc import ABC, abstractclassmethod
from layers import Activation, Flatten, Linear, Norm, BatchNorm, Dropout
from loss import BCE, CCE, SCCE
from optimizer import SGD, SGDM, Adam
from dataset import Dataset
from utils import accuracy, normalize, onehot

class _Net(ABC):

    @abstractclassmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)



class Net(_Net):

    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def add(self, *layers):
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x):
        p = x
        for layer in self.layers:
            p = layer(p)
        return p
        
    def train(self, optimizer, loss, dataset, epochs):

        net_error = 0
        net_acc = 0
        m = len(dataset)

        for epoch in range(epochs):

            accum_error = 0
            accum_acc = 0
            
            for inputs, labels in dataset:

                pred = self.forward(inputs)
                error = loss(pred, labels)
                grad = loss.backward()
                optimizer.update(grad)

                acc = accuracy(pred, labels)
                accum_error += error
                accum_acc += acc
            
            net_error += accum_error
            net_acc += accum_acc
            accum_error /= m
            accum_acc /= m
            print(f"Epoch: {epoch + 1}/{epochs} | Loss: {accum_error:.6f} | Accuracy: {accum_acc * 100:.2f}%")

        avg_error = net_error / epochs
        avg_acc = net_acc / epochs
        return avg_error, avg_acc

    def test(self, loss, dataset):

        net_error = 0
        net_acc = 0
        m = len(dataset)

        for inputs, labels in dataset:
            pred = self.forward(inputs)
            net_error += loss(pred, labels)
            net_acc += accuracy(pred, labels)

        avg_error = net_error / m
        avg_acc = net_acc / m

        print(f"Loss: {avg_error:.4f} | Accuracy: {avg_acc * 100:.2f}%")

        return avg_error, avg_acc
    
    def parameters(self):
        return self.layers


if __name__ == "__main__":
    inputs = np.random.randint(0, 256, (1000, 27, 27, 3))
    inputs = normalize(inputs)
    labels = np.random.choice(3, (1000))
    # labels = onehot(labels)

    dataset = Dataset(inputs, labels, batch_size=64)

    relu = Activation("relu")
    layers = [Flatten(), Linear(27 * 27 * 3, 128), BatchNorm(128), Dropout(0.3), relu,
            Linear(128, 64), BatchNorm(64), Dropout(0.3), relu, Linear(64, 64), BatchNorm(64), 
            Dropout(0.1), relu, Linear(64, 3)]

    loss = SCCE()
    net = Net(*layers)
    optimizer = Adam(net.parameters(), lr=0.01)
    net.train(optimizer, loss, dataset, epochs=50)





