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
        # init weights & bias
        self.params = [np.random.rand(inshape, outshape), 
                    np.zeros((1, outshape))]

    def forward(self, x):
        # find dot product -> return activation of dot product
        self.x = x
        z = np.dot(self.x, self.params[0]) + self.params[1]
        return z

    def backward(self, grads, lr):
        # update params with their gradients
        for param, grad in zip(self.params, grads):
            param -= grad * lr
    
    def gradients(self, grad):
        grads = [np.dot(self.x.T, grad), 
                np.sum(grad, axis=0, keepdims=True)]
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

    def __init__(self, features, gamma=1, beta=0, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.params = [np.zeros((1, features)) + gamma, 
                        np.zeros((1, features)) + beta]

    def forward(self, x: np.ndarray):
        self.x = x
        self.mean = np.mean(x, axis=-1, keepdims=True)
        self.var = np.power(x - self.mean, 2).mean(axis=-1, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (x - self.mean) / self.std
        self.y = self.norm * self.params[0] + self.params[1]
        return self.y

    def backward(self, grads, lr):
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        batch_size, features = self.x.shape
        grads = [np.sum(grad * self.norm, axis=0, keepdims=True), 
                np.sum(grad, axis=0, keepdims=True)]
        grad = self.params[0] * grad
        d_istd = np.sum(grad * (self.x - self.mean), axis=0, keepdims=True)
        d_mean = 1 / self.std * grad
        d_std = -1 / np.power(self.std, 2) * d_istd
        d_var = 0.5 * (1 / self.std) * d_std
        d_sqrt = 1 / batch_size * np.ones((batch_size, features)) * d_var
        d_mean = 2 * d_mean * d_sqrt
        d_mean = -1 * np.sum(d_mean, axis=0, keepdims=True)
        grad = 1 / batch_size * np.ones((batch_size, features)) * d_mean
        return grad, grads

    def learnable(self):
        return True


class BatchNorm(Layer):

    def __init__(self, features, gamma=1, beta=0, eps=1e-9) -> None:
        super().__init__()
        self.eps = eps
        self.params = [np.zeros((1, features)) + gamma, 
                        np.zeros((1, features)) + beta]

    def forward(self, x):
        self.x = x
        self.mean = np.mean(x, axis=0, keepdims=True)
        self.var = np.power(x - self.mean, 2).mean(axis=0, keepdims=True)
        self.std = np.sqrt(self.var + self.eps)
        self.norm = (x - self.mean) / self.std
        self.y = self.params[0] * self.norm + self.params[1]
        return self.y

    def backward(self, grads, lr):
        for param, grad in zip(self.params, grads):
            param -= grad * lr

    def gradients(self, grad):
        batch_size, features = self.x.shape
        grads = [np.sum(grad * self.norm, axis=0, keepdims=True), 
                np.sum(grad, axis=0, keepdims=True)]
        grad = self.params[0] * grad
        d_istd = np.sum(grad * (self.x - self.mean), axis=0, keepdims=True)
        d_mean = 1 / self.std * grad
        d_std = -1 / np.power(self.std, 2) * d_istd
        d_var = 0.5 * (1 / self.std) * d_std
        d_sqrt = 1 / batch_size * np.ones((batch_size, features)) * d_var
        d_mean = 2 * d_mean * d_sqrt
        d_mean = -1 * np.sum(d_mean, axis=0, keepdims=True)
        grad = 1 / batch_size * np.ones((batch_size, features)) * d_mean
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
    relu = Activation("relu")
    softmax = Activation("softmax")
    sigmoid = Activation("sigmoid")
    inlayer = Linear(28 * 28, 8)
    hidlayer = Linear(8, 8)
    outlayer = Linear(8, 3)
    layers = [flatten, inlayer, relu, BatchNorm(8), hidlayer, relu, outlayer]
    optimizer = SGDM(layers, lr=0.1)
    loss = CCE()


    # data & hyperparameter(s)
    inputs = np.random.randint(0, 256, (16, 28, 28, 1))
    labels = np.random.choice(3, (16, ))
    labels = onehot(labels)

    inputs = inputs / 255

    for epoch in range(15):

        x = inputs
        for layer in layers:
            x = layer(x)

        out = x
        error = loss(out, labels)
        print(f"Error: {error}")

        grad = loss.backward()

        optimizer.update(grad)

        
        

    

    

