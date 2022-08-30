import numpy as np

class Perceptron:

    # Initialize the perceptron
    def __init__(self, weights: np.ndarray, bias: float, activation=None) -> None:
        self.weights = weights
        self.bias = bias
        self.activation = activation

    # Weighted Sum of inputs applied to activation function
    def predict(self, inputs: np.ndarray) -> float:
        y = np.dot(inputs, self.weights) + self.bias
        return self.activation(y)
        

# Activation functions
def step(y: float, threshold: float=0) -> int:
    return 1 if y >= threshold else 0

def sigmoid(z: float) -> float:
    return 1 / (1 + np.exp(-z))

def tanh(x: float) -> float:
    return 2 / (1 + np.exp(-2 * x)) - 1

if __name__ == '__main__':
    # AND Gate Example
    AND = Perceptron(np.array([1, 1]), -2, step)
    inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    print('AND Gate:')
    for ins in inputs:
        print(f'{ins} -> {AND.predict(ins)}')

    # OR Gate Example
    OR = Perceptron(np.array([1, 1]), -1, step)
    print('OR Gate:')
    for ins in inputs:
        print(f'{ins} -> {OR.predict(ins)}')
