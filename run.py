from Network import Network
from FullyConnectedLayer import FullyConnectedLayer
import numpy as np

X = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
y = np.array([[0],[1],[1],[0]])

layers = [
    FullyConnectedLayer(X.shape[1], 4, False, 'sigmoid'),
    FullyConnectedLayer(4, 1, True, 'sigmoid')
]

net = Network(8000, layers)
net.train(X, y)