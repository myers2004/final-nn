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
    #Test a known output is given
    epsilon = 0.000001

    arch = [{'input_dim': 8, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(arch, 0.1, 1, 4, 100, 'BCE')
    
    y = np.array([0,1,0,1])
    y_hat = np.array([0.2,0.1,0.9,0.8])

    BCE = test_nn._binary_cross_entropy(y, y_hat)

    assert np.isclose(BCE, 1.2628643, epsilon)


def test_binary_cross_entropy_backprop():
    #Test a known output is given
    epsilon = 0.00001

    arch = [{'input_dim': 8, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(arch, 0.1, 1, 4, 100, 'BCE')
    
    y = np.array([0,1,0,1])
    y_hat = np.array([0.2,0.1,0.6,0.7])

    dBCE = test_nn._binary_cross_entropy_backprop(y, y_hat)

    dBCE_man = np.array([0.3125, -2.5, 0.625, -0.35714286])

    assert np.all(np.isclose(dBCE, dBCE_man, epsilon))


def test_mean_squared_error():
    #Test a known output is given
    epsilon = 0.001

    arch = [{'input_dim': 8, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(arch, 0.1, 1, 4, 100, 'BCE')
    
    y = np.array([0,1,0,1])
    y_hat = np.array([0.2,0.1,0.9,0.8])

    MSE = test_nn._mean_squared_error(y, y_hat)

    assert np.isclose(MSE, 0.425, epsilon)

def test_mean_squared_error_backprop():
    #Test a known output is given
    epsilon = 0.01

    arch = [{'input_dim': 8, 'output_dim': 32, 'activation': 'relu'}, {'input_dim': 32, 'output_dim': 8, 'activation': 'sigmoid'}]
    test_nn = nn.NeuralNetwork(arch, 0.1, 1, 4, 100, 'BCE')
    
    y = np.array([0,1,0,1])
    y_hat = np.array([0.2,0.1,0.6,0.7])

    dMSE = test_nn._mean_squared_error_backprop(y, y_hat)

    dMSE_man = np.array([0.1, -0.45, 0.3, -0.15])

    assert np.all(np.isclose(dMSE, dMSE_man, epsilon))
    

def test_sample_seqs():
    #Create two classes of seqs that are identical within each class
    # Test classes are balenced, have 1000 each, and that labels and classes line up
    pos_seq = 'AAAA'
    pos = []
    for i in range(20):
        pos.append(pos_seq)

    neg_seq = 'BBB'
    neg = []
    for i in range(73):
        neg.append(neg_seq)

    #Create label vectors
    pos_label = [1 for i in range(len(pos))]
    neg_label = [0 for i in range(len(neg))]

    #Two long lists
    seqs = pos + neg
    labels = pos_label + neg_label

    #Balence classes with sample_seqs
    sampled_seqs, sampled_labels = preprocess.sample_seqs(seqs, labels)

    #Count how many of each class are present, and check the label agrees with the seq
    num_pos = 0
    num_neg = 0
    for i in range(len(sampled_labels)):
        if sampled_labels[i] == 1:
            assert sampled_seqs[i] == pos_seq
            num_pos += 1
        else:
            assert sampled_seqs[i] == neg_seq
            num_neg += 1
    
    #Test for balenced class sizes
    assert num_neg == num_pos

    #Test that class size is 1000
    assert num_neg == 1000

def test_one_hot_encode_seqs():
    #Encode a seq with a known output
    seq = ['AGATC']
    encoded_seq = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0]])

    seq = preprocess.one_hot_encode_seqs(seq)

    for i in range(len(seq[0])):
        assert seq[0][i] == encoded_seq[0][i]

    #Make sure length of encoded is 4x as long for more seqs
    seqs = ['AGATC', 'TGACA', 'ACTGA', 'AAAAA', 'TAGAC']

    seqs = preprocess.one_hot_encode_seqs(seqs)
    for seq in seqs:
        assert len(seq) == 5 * 4

    #Try to pass an unallowed base
    bad_seq = ['Sean']
    with pytest.raises(KeyError):
        preprocess.one_hot_encode_seqs(bad_seq)
        