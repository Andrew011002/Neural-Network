from abc import abstractclassmethod, ABC
import numpy as np
from activations import ReLU, Sigmoid, Softmax, Tanh
from loss import BCE, MSE, CCE, SCCE, NLL
from utils import onehot, unhot
from optimizer import SGDM, SGD, Adam



class Layer(ABC):

    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractclassmethod
    def forward(self, *args):
        pass
    
    @abstractclassmethod
    def backward(self, *args):
        pass

    @abstractclassmethod
    def learnable(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Linear(Layer):

    def __init__(self, inshape, outshape):
        super().__init__()
        # init weights, bias & activation
        self.params = [np.random.rand(inshape, outshape) - 0.5,
                        np.zeros((1, outshape))]

    def forward(self, x):
        # find dot product -> return activation of dot product
        self.x = x
        z = np.dot(self.x, self.params[0]) + self.params[1]
        return z

    def backward(self, grads, lr):
        # find local grad -> find delta for weights -> avg grad & update -> return new grad
        for param, grad in zip(self.params, grads):
            param -= grad * lr
    
    def gradients(self, grad):
        grads = [np.dot(self.x.T, grad), 
                np.dot(np.ones((len(self.x), 1)).T, grad)]
        grad = np.dot(grad, self.params[0].T)
        return grad, grads

    def learnable(self):
        return True


class Activation(Layer):
    
    def __init__(self, activation):
        super().__init__()
        # init activation form activation maps
        activations = dict(relu=ReLU, softmax=Softmax, sigmoid=Sigmoid, tanh=Tanh)
        self.activation = activations[activation]()
        self.inputs = []

    def forward(self, z):
        # set input -> apply & return activation
        self.inputs.append(z)
        return self.activation(z)

    def backward(self, grad):
        # calculate grad by derivative of activation times og grad
        grad = self.activation.derivative(self.inputs.pop()) * grad
        return grad

    def learnable(self):
        return False


class Norm(Layer):

    def __init__(self, inshape, gamma=1, beta=0, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.params = [np.zeros((1, inshape)) + gamma, 
                        np.zeros((1, inshape)) + beta]

    def forward(self, a: np.ndarray):
        self.a = a
        mean = np.mean(self.a, axis=-1, keepdims=True)
        var = np.power(self.a - mean, 2)
        std = np.sqrt(var + self.eps)
        norm = (self.a - mean) / std
        return norm * self.params[0] + self.params[1]

    def backward(self, grads, lr):
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        # TODO
        grads = [None, None]
        return grad, grads

    def learnable(self):
        return True


class BatchNorm(Layer):

    def __init__(self, gamma=1, beta=0, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.params = [gamma, beta]

    def forward(self, b):
        self.b = b
        mean = np.mean(self.b, axis=0, keepdims=True)
        var = np.power(self.b - mean, 2)
        std = np.sqrt(var + self.eps)
        norm = (self.b - mean) / std
        return norm * self.params[0] + self.params[1]

    def backward(self, grads, lr):
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        # TODO
        grads = [None, None]
        return grad, grads

    def learnable(self):
        return True


class Dropout(Layer):
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x):
        features = x.shape[1]
        n = int(features * self.p)
        drops = np.random.choice(features, n, replace=False)
        x[:, drops] = 0
        return x
        
    def backward(self, grad):
        return grad

    def learnable(self):
        return False


class Flatten(Layer):
    
    def forward(self, x):
        super().__init__()
        # reshape to (batch_size, left over dimensions)
        return x.reshape((x.shape[0], -1))

    def backward(self, grad):
        # no gradient just return the gradient
        return grad

    def learnable(self):
        return False


if __name__ == '__main__':
    # layers
    flatten = Flatten()
    dropout = Dropout(0.3)
    norm = BatchNorm()
    relu = Activation("relu")
    softmax = Activation("softmax")
    inlayer = Linear(28 * 28, 8)
    hidlayer = Linear(8, 8)
    outlayer = Linear(8, 3)
    layers = [flatten, inlayer, dropout, relu, hidlayer, relu, outlayer]
    optimizer = Adam(layers, lr=0.01)
    loss = CCE()


    # data & hyperparameter(s)
    inputs = np.random.randint(0, 256, (16, 28, 28, 1))
    labels = np.random.choice(3, (16, ))
    labels = onehot(labels)

    inputs = inputs / 255

    for epoch in range(10):

        x = inputs
        for layer in layers:
            x = layer(x)

        out = x
        error = loss(out, labels)
        print(f"Error: {error}")

        grad = loss.backward()

        optimizer.update(grad)

        
        

    

    

