# Imports
import numpy as np
from typing import List, Tuple
from numpy.typing import ArrayLike
import random

def sample_seqs(seqs: List[str], labels: List[bool]) -> Tuple[List[str], List[bool]]:
    """
    This function should sample the given sequences to account for class imbalance. 
    Consider this a sampling scheme with replacement.
    
    Args:
        seqs: List[str]
            List of all sequences.
        labels: List[bool]
            List of positive/negative labels

    Returns:
        sampled_seqs: List[str]
            List of sampled sequences which reflect a balanced class size
        sampled_labels: List[bool]
            List of labels for the sampled sequences
    """
    #First let's find how many of each type we have
    pos_label_count = labels.count(True)
    neg_label_count = labels.count(False)

    #We want the max number of data points, while at the same time having balenced class sizes
    # To acheive this, randomly sample with replacement from both classes the same number of times

    #Get index of true and false postions
    pos_index = [] 
    neg_index = []
    for i in len(labels):
        if labels[i] == True:
            pos_index.append(i)
        else:
            neg_index.append(i)

    sampling_rounds = 200

    sampled_seqs = []
    sampled_labels = []

    #Sample both postive and negative with replacement
    sampled_pos_idxs = np.random.choice(pos_index, sampling_rounds, replace= True)
    sampled_neg_idxs = np.random.choice(neg_index, sampling_rounds, replace= True)


    #Add the sampled data points to the sampled lists
    for idx in sampled_pos_idxs:
        sampled_seqs.append(seqs[idx])
        sampled_labels.append(labels[idx])
    for idx in sampled_neg_idxs:
        sampled_seqs.append(seqs[idx])
        sampled_labels.append(labels[idx])

    #Shuffle the sampled seqs, maintaining the same order across sampled_seqs and sampled_labels
    # to prevent any bias in order
    zipped = list(zip(sampled_seqs, sampled_labels))
    random.shuffle(zipped)
    sampled_seqs, sampled_labels = zip(*zipped)

    return sampled_seqs, sampled_labels


def one_hot_encode_seqs(seq_arr: List[str]) -> ArrayLike:
    """
    This function generates a flattened one-hot encoding of a list of DNA sequences
    for use as input into a neural network.

    Args:
        seq_arr: List[str]
            List of sequences to encode.

    Returns:
        encodings: ArrayLike
            Array of encoded sequences, with each encoding 4x as long as the input sequence.
            For example, if we encode:
                A -> [1, 0, 0, 0]
                T -> [0, 1, 0, 0]
                C -> [0, 0, 1, 0]
                G -> [0, 0, 0, 1]
            Then, AGA -> [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0].
    """

    encodings = []

    allowed_bases = ['A', 'T', 'C', 'G']

    one_hot_key = {'A' : [1,0,0,0],
                   'T' : [0,1,0,0],
                   'C' : [0,0,1,0],
                   'G' : [0,0,0,1]}

    for base in seq_arr:
        if base not in allowed_bases:
            raise(KeyError('Non allowed base' + base + 'in sequences'))
        encodings.extend(one_hot_key[base])

    return encodings