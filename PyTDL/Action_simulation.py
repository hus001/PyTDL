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

# 忽略特定类型的警告，例如 DeprecationWarning
warnings.filterwarnings("ignore")


def get_genAction_1(df,iter_num):

    stimu_df_list = []

    for iter_i in range(iter_num):
        # 随机选择动作序列和计算奖励
        # np.random.seed(400)  # 设置随机种子以确保结果可重复
        action_sequence = []
        stim_outcome = []  # 对选择进行编码
        motivation = [] # 对选择动机进行编码

        # 随机选择动作序列和计算奖励
        for _, row in df.iterrows():
            p_lick = row['P_lick']
            p_nolick = row['P_nolick']
            
            # 根据概率值选择CP或CG动作
            choice = np.random.choice(['lick', 'no_lick'], p=[p_lick, p_nolick])
            
            if choice == 'lick':
                st_outcome = 1  # lick
                motiv = 1
            elif choice == 'no_lick':
                st_outcome = 2  # no-lick
                motiv = 0
            
            action_sequence.append(choice)
            stim_outcome.append(st_outcome)
            motivation.append(motiv)
        
        # 创建一个新的DataFrame来存储这些变量的值
        stimu_df = pd.DataFrame({'action_sequence': action_sequence, 'stim_outcome': stim_outcome, 'motivation': motivation})
        stimu_df['iter_num'] = iter_i+1
        stimu_df_list.append(stimu_df)

    return stimu_df_list


def get_genAction(df,iter_num):

    stimu_df_list = []

    for iter_i in range(iter_num):
        # 随机选择动作序列和计算奖励
        # np.random.seed(400)  # 设置随机种子以确保结果可重复
        action_sequence = []
        stim_outcome = []  # 对选择进行编码
        motivation = [] # 对选择动机进行编码

        # 随机选择动作序列和计算奖励
        # print(result_df)
        for _, row in df.iterrows():
            p_cp = row['P_CP']
            p_cg = row['P_CG']
            
            # ε-greedy策略
            if np.random.rand() < row['epsilon']:
                # 随机选择CP、CG或Miss动作
                choice = np.random.choice(['CP', 'CG'], p=[1/2, 1/2])

            else:
                # 根据概率值选择CP或CG动作
                choice = np.random.choice(['CP', 'CG'], p=[p_cp, p_cg])
            
            if choice == 'CP':
                st_outcome = 1  # Miss选择的奖励值为0
                motiv = 1
            elif choice == 'CG':
                st_outcome = 2  # CP选择的奖励值为+1
                motiv = -1
            
            action_sequence.append(choice)
            stim_outcome.append(st_outcome)
            motivation.append(motiv)
        
        # 创建一个新的DataFrame来存储这些变量的值
        stimu_df = pd.DataFrame({'action_sequence': action_sequence, 'stim_outcome': stim_outcome, 'motivation': motivation})
        stimu_df['iter_num'] = iter_i+1
        stimu_df_list.append(stimu_df)

    return stimu_df_list

