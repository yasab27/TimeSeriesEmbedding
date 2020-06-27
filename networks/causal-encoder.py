#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Provides CausalCNN and CausalCNNEncoder classes to enable users to generate robust representations of 
multivariate time series data. 

The following method is based on the paper from Franceschi, Dieuleveut, and Jaggi (2020) and published in 
NeurIPS (https://arxiv.org/pdf/1901.10738v4.pdf). The general idea is that through the use of casually diluted
convolution operation inserted into an encoder coupled with a unique negative-sampling based triplet loss, 
one can generate robust representations of time series without labels and of customizable dimension. Here 
we implement their model as well as some PyTorch helper classes to help in management of the tensor sizes. The 
details of their models are described in the associated readme.

Adapated from https://bit.ly/3g49cPj 
"""

import torch # High-Level library for constructing deep ANN architectures

__author__ = "Yasa Baig"
__license__ = "MIT"
__version__ = "1.0.0"
__maintainer__ = "Yasa Baig"
__email__ = "yasa.baig@duke.edu"
__status__ = "Development"

"""HELPER CLASSES

Here we define custom torch.nn modules which are not associated with any gradients, allow for easy processing of the 
data intermediary stages. Namely we define Chomp1D, a method for restricting our convolutional feature maps to only the
causal information and then also squeeze, which simply provides a wrapper around the native torch tensor squeeze method which 
is compatible with the torch Sequential API.
"""

class Chomp1D(torch.nn.Module):
    """
    A torch module compatible with the Sequential API which eliminates--"chomps"--the last number of elements in a torch tensor.

    The purpose of this class is to provide an easy way to drop the "non-casual" components left over from a 1D casual
    convolutional map of a multivariate time series by simplying eliminating them. Takes as input a rank 3 tensor
    of dimensions (B,C,L) where B is the batch size, C is the number of input channels, and L is the length of the feature
    map (time series representation). This will return a tensor of size (B,C,L-s) where s is the number of entries
    to remove from the length or "chomp_size".

    Attributes
    ----------
    chomp_size: int
        The number of entries to remove from the end of the tensor. 

    Methods
    -------
    forward(X)
        Returns the post-chomped version of rank 3 tensor with the dimension (B,C,L-chomp_size)
    """

    def __init__(self, chomp_size):
        """Initialize a new Chomp1D module.

        Parameters
        ----------
        chomp_size: int, required
            The number of entries to remove from the end of the tensor.
        """

        self.chomp_size = chomp_size
    
    def foward(self,X):
        """Returns the post-chomped version of rank 3 tensor with the dimension (B,C,L-chomp_size)

        Parameters
        ----------
        X: tensor, required
            The rank 3 tensor of dimension (B,C,L) to chomp. 

        Returns
        -------
        chomped_X: tensor
            The rank 3 tensor of dimension (B,C,L-s) after chomping
        """

        chomped_X = X[:,:,:-self.chomp_size]

        return chomped_X



    