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
            A rank 3 tensor containing the output of the convolutional block of dimension (B,C_out,L).
        """

        # First perform the forward propogation from the main block
        main_output = self.causal(input=X)

        # If the input and output channels do not match perform convolution on the residual skip
        # connection. Otherwise simply pass it forward. 
        if self.resid_up_down_sample is not None:
            resid = self.resid_up_down_sample(X)
        else:
            resid = X
        
        # Now combine the outputs and residual connetion through a simple sum
        feature_maps = resid+main_output

        # If a final activation is requested, pass this through one last RElU, otherwise return. 
        if self.use_final_activation:
            return torch.nn.ReLU(feature_maps)
        else:
            return feature_maps


class CausalCNN(torch.nn.Module):
    """
    A network of repeated causal convolution blocks used to extract useful feature maps from 
    multivariate time series data. 

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C_out`, `L`).

    This network appends causal blocks with exponentially increasing size of the dilations based on 
    the prespecified depth of the network. This enables heirarchical integration of time-series-data
    without the need for hidden states or other forms of recurrence. 

    Attributes
    ----------
    network: Sequential
        The torch.nn.Sequential object used to perform a forward pass through the network. This is the main
        network architecture. 
    
    Methods
    -------
    forward(X)
        Takes input rank 3 tensor X of dimension (B,C,L) where B is batch number, C is number of channels, 
        and L is the length and returns tensor of dimension (B,C_out,L) which consists of C_out feature
        maps generated from casual diluted convolution.
    """

    def __init__(self, in_channels, out_channels, inter_channels, depth,  kernel_size):
        """Intialize a new CausalCNN class with the specified hyperparameters. 

        Parameters
        ----------
        in_channels: int
            The number of input channels in the initial time series. This usually corresponds to multivariate 
            time recordings of different varaibles (say position, velocity, acceleration).
        
        out_channels: int
            The number of final desired feature maps in the output.
        
        inter_channels: int
            The number of intermediate channels utilized within the sequence of casually diluted block connections.

        depth: int
            The number of causal blocks to utilize

        kernel_size: int
            The size of the 1D kernel utilized in the convolutional blocks. 
        """
        super(CausalCNN, self).__init__()

        layers = [] # Used to store the convolutional blocks
        dilation_size = 1 # Initial dilation size. 

        # Generate all of the causal blocks iteratively with increasing dilution size
        for i in range(depth):
            
            in_channels_block = 0 # Stores the in_channels for the ith convolutional block

            # If this is the first block, ensure that it has input number of channels equal to 
            # in_channels
            if i == 0:
                in_channels_block = in_channels
            else:
                in_channels_block = inter_channels
            
            new_block = CausalConvolutionBlock(
                in_channels=in_channels_block,
                out_channels =inter_channels,
                kernel_size=kernel_size,
                dilation = dilation_size,
                use_final_activation=False
            )

            layers.append(new_block)

            # Double the dilation size by 2 to ensure exponentially weighted dilation for the next layer
            dilation_size *= 2
        
        
        # Append the last layer seperately in order to make sure the number of final channels matches 
        # the desired number. 
        layers.append(CausalConvolutionBlock(
            in_channels = inter_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            dilation = dilation_size
        ))

        # Lastly generate a sequential object to store all of the layers. 
        self.network = torch.nn.Sequential(*layers)

    def forward(self,X):
        """Given an input rank 3 tensor X of dimensions (B,C,L), return a convolved representation of the
        tensor.

        Parameters
        ----------
        X: tensor        
            A rank 3 tensor of dimension (B,C,L) where B is batch size, C is the number of channels,
            and L is the length of the sequences.

        Returns
        -------
        feature_maps: tensor
            A rank 3 tensor containing the output of the convolutional block network of dimension (B,C_out,L).
        """

        feature_maps = self.network(X)
        return feature_maps


class CausalCNNEncoder(torch.nn.Module):
    """
    Encoder of a time series using a causal CNN: the computed representation is
    the output of a fully connected layer applied to the output of an adaptive
    max pooling layer applied on top of the causal CNN, which reduces the
    length of the time series to a fixed size.

    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`).

    Attributes
    ----------
    network: torch.nn.Sequential
        The internal feed forward network of the encoder. 
    
    Methods
    -------
    forward(X)
        Returns the representations of a time series X
    """

    def __init__(self, in_channels, inter_channels, depth, reduced_size,out_channels, kernel_size):
        """Initialize a new Causal CNN class with the given hyperparameters

        Parameters
        ----------
        in_channels: int
            The number of input channels of the time series to be compressed. Note that this will be equal to the 
            number of variables measured. 
        
        inter_channels: int
            The number of internal channels to use within the causal blocks when computing the representation
        
        depth: int
            The number of causal blocks to utilize when computing the time series. 
        
        reduced_size:
            The number of desired output channels from the encoder portion of the network

        out_channels:
            The number of final output channels desired after the linear layer

        kernel_size:
            Size of 1D kernel to utilize in convolutional steps.
        """
        super(CausalCNNEncoder, self).__init__()

        # Generate a causal CNN layer with the required hyperparams
        causal_cnn = CausalCNN(
            in_channels = in_channels,
            out_channels= reduced_size,
            inter_channels = inter_channels,
            depth = depth, 
            kernel_size = kernel_size
        )

        # Reduce the size utilizing max pooling and then squeeze the dimension
        reduce_size = torch.nn.AdaptiveMaxPool1d(1)
        squeeze = SqueezeChannels()  # Squeezes the third dimension (time)

        # Lastly perform one linear transformation to get the number of desired output channels. 
        linear = torch.nn.Linear(reduced_size, out_channels)

        # Concatenate all operations into this network. 
        self.network = torch.nn.Sequential(
            causal_cnn, reduce_size, squeeze, linear
        )

    def forward(self, X):
        """Given a multivariable time series, return a representation

        Parameters
        ----------
        X: tensor
            The multivariate time series input fed to the network. Must be rank 3 tensor. 

        Returns
        -------
        representation: tensor
            A rank 2 tensor of the computing time series representaiton
        """

        representation = self.network(X)
        return representation









    