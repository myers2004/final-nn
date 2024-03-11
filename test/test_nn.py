# TODO: import dependencies and write unit tests below
#import pacakges
import numpy as np
from nn import (nn, io, preprocess)
import pytest

def test_single_forward():
    #Test feed foward on a toy input, modifiying wights and biases
    # manually
    epsilon = 0.000001

    #Set up dummy variables
    X = np.array([[1,2],[4,4]])
    X = X.T
    arch = [{'input_dim': 8, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(arch, 0.1, 1, 2, 100, 'BCE')

    W_curr = np.array([[0.5,0.4],[0.1,0.2]])
    b_curr = np.array([[0.1],[0]])

    Z_man = np.array([[1.4,3.7],[0.5,1.2]])
    A_man = test_nn._sigmoid(Z_man)

    #Test with sigmoid
    A_man = test_nn._sigmoid(Z_man)

    Z_curr, A_curr = test_nn._single_forward(W_curr,b_curr,X, 'sigmoid')
    assert np.all(np.isclose(Z_curr, Z_man,epsilon))
    assert np.all(np.isclose(A_curr, A_man, epsilon))

    #Test with relu
    A_man = test_nn._relu(Z_man)

    Z_curr, A_curr = test_nn._single_forward(W_curr,b_curr,X, 'relu')
    assert np.all(np.isclose(Z_curr, Z_man,epsilon))
    assert np.all(np.isclose(A_curr, A_man, epsilon))

    #Test with unallowed activation function

    with pytest.raises(KeyError):
        Z_curr, A_curr = test_nn._single_forward(W_curr,b_curr,X, 'junk')






def test_forward():
    pass

def test_single_backprop():
    pass

def test_predict():
    pass

def test_binary_cross_entropy():
    pass

def test_binary_cross_entropy_backprop():
    pass

def test_mean_squared_error():
    pass

def test_mean_squared_error_backprop():
    pass

def test_sample_seqs():
    pass

def test_one_hot_encode_seqs():
    pass