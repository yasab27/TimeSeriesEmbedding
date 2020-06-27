#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    Provides CausalCNN and CausalCNNEncoder classes to enable users to generate robust representations of 
    multivariate time series data. 

    The following method is based on the paper from Franceschi, Dieuleveut, and Jaggi (2020) and published in 
    NeurIPS (https://arxiv.org/pdf/1901.10738v4.pdf). The general idea is that through the use of casually diluted
    convolution operation inserted into an encoder coupled with a unique negative-sampling based triplet loss, 
    one can generate robust representations of time series without labels and of customizable dimension. Here 
    we implement their model as well as some PyTorch helper classes to help in management of the tensor sizes. The 
    details of their models are described in the associated readme. 
"""
