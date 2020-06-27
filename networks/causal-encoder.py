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

        super(Chomp1D, self).__init__()
        self.chomp_size = chomp_size
    
    def forward(self,X):
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

        chomped_X = X[:,:,:-self.chomp_size] # Keep observations and the number of channels the same.

        return chomped_X

class SqueezeChannels(torch.nn.Module):
    """
    A torch module compatible with the Sequential API which squeeze a 3D tensor down to 2D.

    This is utilized in the end of the model in before our final linear transformation down to the desired
    fixed size of our representations.


    Methods
    -------
    forward(X)
        Returns the squeezed version of the rank 3 tensor X down to two dimensions
    """

    def __init__(self):
        """Initialize a new SqueezeChannels instance. Takes no parameters."""

        super(SqueezeChannels, self).__init__()

    def forward(self,X):
        """Compute and return the squeezed version of the input tensor X.

        Parameters
        ----------
        X: tensor
            A rank 3 tensor to be squeezed down to two dimensions.

        Returns
        -------
        squeezed_X: tensor
            A rank 2 tensor where the third dimension (generally time) has been squeezed. 
        """

        squeezed_X = X.squeeze(2) # The 2D rank reduction here is hard coded.

        return squeezed_X

"""NETWORK CLASSES

We now define the artifical neural network architecture for the causal blocks, the casual CNN, and then the causal
encoder.
"""

class CausalConvolutionBlock(torch.nn.Module):
    """
    A diluted causal convolution block consisting of two sequential diluted causal layers with leaky relu 
    activation functions. 
    
    Each layer increases in dilution rate exponentially to better integrate heirarchical relationships in 
    the time series data. Additionally, a parallel residual skip connection
    between the first and last layer helps avoid vanishing gradient problem as well as integrates more
    heirarchical data into later layers. Weight normalization is utilized in the convolutional layers 
    to accelerate training and prevent explosion of weights (i.e. an alternative to L2 reg.)

    Takes as input a rank 3 tensor of dimension (B,C,L) where B is batch size, C is the number of channels,
    and L is the length of the sequences. The output is also a rank 3 tensor but with dimensions (B,C_out,L),
    meanign that the length of the sequences remain the same but potentially more feature maps are generated. 

    Attributes
    ----------
    causal: torch.nn.Sequential
        The Sequential object encoding the forward propogation architecture of the network. 

    resid_up_down_sample: torch.nn.Conv1d
        A conv1d operation which is applied to ensure that the number of channels in the 
        skip connection match before adding them to the output of the main forward propogation. If
        the channel numbers already match, then this is ignored. 

    use_final_activation: boolean
        Boolean encoding whether or not to apply one final ReLU activation function to the output
        after summing the main and skip connection. (Default is False)

    Methods
    -------
    forward(X):
        Returns a tensor of dimensions (B,C_out,L) which is a nonlinear, convolved collection of feature
        maps of the input tensor. 
    """

    def __init__(self,in_channels, out_channels, kernel_size, dilation, use_final_activation=False):
        """Initialize a new causal convolutional block with specific channel numbers, kernal size, and dilation
        parameters. 

        Parameters
        ----------
        in_channels: int
            The number of channels in the input tensor to be convolved. If this is the initial convolutional block
            this corresponds to the individual variables of a multivariable time series (i.e. [position, speed, acc]).
        
        out_channels: int
            The number of desired channels (feature maps) we want the convolutional blocks to generate. 
        
        kernel_size: int
            The length of the 1D kernel utilized for building our filters. This effectively corresponds to length
            of the sliding window slid left-to-right over the course of convolution.
        
        dilation: int
            The dilation factor to consider while performing convolution.
        
        use_final_activation: bool
            Whether or not to perform one final activation function on the channels before passing them to the
            next step of the network. By default this is set to false. 
        """
        
        super(CausalConvolutionBlock,self).__init__()

        # Begin by first computing the padding size for the 1D convolution and later causal truncation
        padding = (kernel_size-1)*dilation

        # Create first causal convolution stage. This consists of a weight normalized regular 1D convolution
        # which is then truncated to only retain the causal information (i.e. ignore the last 'padding' entries)
        # and then passed through a leaky ReLU activation function.
        
        # Convolve the input
        convolution_layer_one = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )

        # Truncate non-casual section
        chomp_layer_one = Chomp1D(padding)

        # Pass through Leaky ReLU activation
        activation_one = torch.nn.LeakyReLU

        # Now create the second convolutional block. This one functions identically save that the input channels
        # to this block will be equal to the output channels of the previous block. The number of output channels
        # from this block will be equal to the number of desired outputs. 

        convolution_layer_two = torch.nn.utils.weight_norm(
            torch.nn.Conv1d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation
            )
        )

        # Truncate non causal section
        chomp_layer_two = Chomp1D(padding)

        # Pass through activation function
        activation_two = torch.nn.LeakyReLU

        # Now string all of these layers into one sequential object
        self.causal = torch.nn.Sequential(
            convolution_layer_one,
            chomp_layer_one,
            activation_one,
            convolution_layer_two,
            chomp_layer_two,
            activation_two
        )

        # Now configure residual connection. If the number of of our input and output channels
        # do not match then we perform a seperate parallel 1D convolution operation. 
        if in_channels != out_channels:

            self.resid_up_down_sample = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1 # We use a kernel size of one to ensure the dimensions match in length of our main path. 
            )

        else:
            
            self.resid_up_down_sample = None

    def forward(self,X):
        """Given an input of multivariate time series feature map X, return a new feature map generated by convolution

        Parameters
        ----------
        X: tensor
            A rank 3 tensor of dimension (B,C,L) where B is batch size, C is the number of channels,
            and L is the length of the sequences.

        Returns
        -------
        feature_maps: tensor
            A rank 3 tensor containing the output of the convolutional block.
        """








    