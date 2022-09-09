from abc import ABC, abstractclassmethod
from dataset import Dataset
from loss import Loss
from modules import Module
from optimizers import Optimizer
from utils import accuracy


class Model(ABC):

    """
    Abstract Class for all Models
    """

    @abstractclassmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Sequential(Model):

    """
    Fully Connected Feed-Forward Neural Network
    """

    def __init__(self, *layers: Module):
        # init layers if exist
        super().__init__()
        self.layers = list(layers)

    def add(self, *layers: Module):
        # add layer to layers
        for layer in layers:
            self.layers.append(layer)

    def forward(self, x):
        p = x
        # apply inputs to layers
        for layer in self.layers:
            p = layer(p)
        return p
        
    def train(self, optimizer: Optimizer, loss: Loss, dataset: Dataset, epochs=3):
        # prepare net metrics
        net_error = 0
        net_acc = 0
        m = len(dataset)
        # train over epochs
        for epoch in range(epochs):
            # accumulate metrics over epochs
            accum_error = 0
            accum_acc = 0
            # update over batches
            for inputs, labels in dataset:
                # calc pred -> calc error -> calc grad -> update
                pred = self.forward(inputs)
                error = loss(pred, labels)
                grad = loss.backward()
                optimizer.update(grad)
                # calc acc -> update accumulative metrics
                acc = accuracy(pred, labels)
                accum_error += error
                accum_acc += acc
            # update net metrics -> average accumulative metrics
            net_error += accum_error
            net_acc += accum_acc
            accum_error /= m
            accum_acc /= m
            # display info
            print(f"Epoch: {epoch + 1}/{epochs} | Loss: {accum_error:.6f} | Accuracy: {accum_acc * 100:.2f}%")
        # avg metrics
        avg_error = net_error / epochs
        avg_acc = net_acc / epochs
        return avg_error, avg_acc

    def test(self, loss: Loss, dataset: Dataset):
        # prepare net metrics
        net_error = 0
        net_acc = 0
        m = len(dataset)
        # test over batches
        for inputs, labels in dataset:
            pred = self.forward(inputs)
            net_error += loss(pred, labels)
            net_acc += accuracy(pred, labels)
        # avg metrics
        avg_error = net_error / m
        avg_acc = net_acc / m
        # display infp
        print(f"Loss: {avg_error:.4f} | Accuracy: {avg_acc * 100:.2f}%")
        return avg_error, avg_acc
    
    def parameters(self):
        # give back all layers
        return self.layers