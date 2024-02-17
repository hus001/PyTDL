# MIT License

# Copyright (c) 2023 Qiyun Wu

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import glob 
import os
from scipy.stats import rankdata
from scipy.signal import savgol_filter
# 忽略特定类型的警告，例如 DeprecationWarning
warnings.filterwarnings("ignore")


def softmax_update(Q,Scale_Factor):

    scale_factor=Scale_Factor
    
    e_Q = np.exp(scale_factor * (Q - np.max(Q)))
    P = e_Q / e_Q.sum(axis=1, keepdims=True)

    return P


# 定义带有水平偏移参数的递减Sigmoid函数
def shifted_decreasing_sigmoid(x, steepness, x_offset):
    return 1 - 1 / (1 + np.exp(-(x - x_offset) * steepness))



def softmax_update_starev(Qstable, stable_Sfactor,Qreverse, reverse_Sfactor):

    e_Q1 = np.exp(stable_Sfactor * (Qstable - np.max(Qstable)))
    P1 = e_Q1 / e_Q1.sum(axis=1, keepdims=True)

    e_Q2 = np.exp(reverse_Sfactor * (Qreverse - np.max(Qreverse)))
    P2 = e_Q2 / e_Q2.sum(axis=1, keepdims=True)
    
    P = np.concatenate((P1, P2), axis=0)

    return P