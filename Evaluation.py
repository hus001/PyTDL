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
# 忽略特定类型的警告
warnings.filterwarnings("ignore")


# 累计决策趋势一致性分析相关函数
def get_person_factor(sequence_1,sequence_2):
    sequence_x = sequence_1
    sequence_y = sequence_2
    # 计算均值
    mean_x = np.mean(sequence_x)
    mean_y = np.mean(sequence_y)
    # 计算皮尔逊相关系数
    numerator = np.sum((sequence_x - mean_x) * (sequence_y - mean_y))
    denominator_x = np.sqrt(np.sum((sequence_x - mean_x) ** 2))
    denominator_y = np.sqrt(np.sum((sequence_y - mean_y) ** 2))
    correlation_coefficient = numerator / (denominator_x * denominator_y)

    return correlation_coefficient


def spearman_rank_correlation(x, y):
    """
    计算两个序列的斯皮尔曼秩相关系数
    参数:
    x, y (array-like): 两个待比较的序列,可以是NumPy数组、列表或其他可迭代对象。
    返回: rho (float): 斯皮尔曼秩相关系数的值，范围在-1到1之间,越接近1表示越强的正相关性,越接近-1表示越强的负相关性。
    """
    # 使用rankdata函数将序列转换为秩
    ranks_x = rankdata(x)
    ranks_y = rankdata(y)
    # 计算秩的差值
    rank_diff = ranks_x - ranks_y
    # 计算斯皮尔曼秩相关系数
    n = len(x)
    rho = 1 - (6 * np.sum(rank_diff**2)) / (n * (n**2 - 1))

    return rho


# 滑动窗口相关函数 验证数理特性的一致性===============
def calculate_probability(sub_df,mode):
    '''
    mode:
    0:计算实验数据的概率    
    1:计算仿真数据的概率 
    '''
    if mode == 0:
        # 如果是实验数据,则滑动窗口截取go-nogo所有实验,统计go信号下的lick rate
        sub_df = sub_df[sub_df['stimulus']==1]
        total_rows = len(sub_df)
        outcome_1_rows = len(sub_df[sub_df['outcome'] == 1]) + len(sub_df[sub_df['outcome'] == 4])
    else:
        outcome_1_rows = len(sub_df[sub_df['stim_outcome'] == 1])
        total_rows = len(sub_df)
    probability = outcome_1_rows / total_rows

    return probability

def slide_window_probabilities(df,windowSize,stepSize,Mode):

    window_size = windowSize  # 滑动窗口大小
    step_size = stepSize  # 步长

    probabilities = []  # 存储计算得到的概率值

    for i in range(0, len(df) - window_size + 1, step_size):
        sub_df = df.iloc[i:i+window_size]

        probability = calculate_probability(sub_df,Mode)

        probabilities.append(probability)

    return probabilities


# 基于编辑距离的序列相似性分析相关函数===================
def edit_distance(str1, str2):
    # 计算编辑距离
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                cost = 0
            else:
                cost = 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    edit_dist = dp[m][n]

    return edit_dist


def similarity_from_edit_distance(edit_distance, max_length):
    # 计算相似性指标
    similarity = 1 - (edit_distance / max_length)

    return similarity